"""Evaluate a NeoForce checkpoint on the held-out OpenNeoData test split.

Reports three metrics:

* **MAE / RMSE** of the predicted force field, over ground-truth contact pixels, per
  channel and averaged. Both are **relative**: the dataset divides the target by each
  channel's training scale, and the error is left on that scale rather than converted
  back. The numbers are therefore unit-free and comparable across tactile sensors and
  output resolutions whose raw force scales differ by orders of magnitude.
* **mIoU** of the contact mask, per channel and averaged, at a fixed probability
  threshold.

Contact pixels are the reference for the force metrics because the ground truth is
exactly zero elsewhere: averaging over the whole map would mostly measure how much of
the (mostly empty) background the model got right.

    python -m neoforce.evaluate \\
      --checkpoint outputs/neoforce/neoforce_last.pt \\
      --config configs/neoforce.json \\
      --data-root /path/to/OpenNeoData \\
      --split splits/test.jsonl
"""

import argparse
import json
from pathlib import Path
from typing import Dict, Optional

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from neoforce.data import CHANNELS, ForceNormalizer, OpenNeoDataConfig, OpenNeoDataset
from neoforce.models import NeoForceConfig, NeoForceEncoder


def force_norm_from_checkpoint(ckpt: Dict) -> Optional[ForceNormalizer]:
    """Recover the training-time force scale, preferring the explicit record."""
    normalizer = ForceNormalizer.from_dict(ckpt.get("force_norm"))
    if normalizer is not None:
        return normalizer
    data_cfg = ckpt.get("data_cfg") or {}
    scale = data_cfg.get("force_std")
    if not scale:
        return None
    return ForceNormalizer(scale=tuple(scale), mean=tuple(data_cfg.get("force_mean") or ()), source="checkpoint.data_cfg")


def load_model(checkpoint: str, device: torch.device):
    ckpt = torch.load(checkpoint, map_location="cpu")
    known = NeoForceConfig.__dataclass_fields__
    model_cfg = NeoForceConfig(**{k: v for k, v in ckpt.get("model_cfg", {}).items() if k in known})
    model = NeoForceEncoder(model_cfg)
    result = model.load_state_dict(ckpt["model"] if "model" in ckpt else ckpt, strict=False)

    # A decoder loaded at its random init would still produce plausible-looking metrics.
    missing_decoder = [k for k in result.missing_keys if k.startswith("decoder.")]
    if missing_decoder:
        raise SystemExit(
            f"Decoder weights missing from the checkpoint: {missing_decoder[:6]}. The "
            "checkpoint and the model config disagree, so the metrics would be meaningless."
        )
    print(f"loaded {Path(checkpoint).name} (step {ckpt.get('step')}), "
          f"{len(result.unexpected_keys)} training-only tensors ignored")
    model.to(device).eval()
    return model, model_cfg, force_norm_from_checkpoint(ckpt)


class Metrics:
    """Streaming accumulators, so the whole split never has to be held in memory."""

    def __init__(self, device: torch.device, channels: int = 6):
        self.abs_err = torch.zeros(channels, device=device)
        self.sq_err = torch.zeros(channels, device=device)
        self.contact = torch.zeros(channels, device=device)
        self.intersection = torch.zeros(channels, device=device)
        self.union = torch.zeros(channels, device=device)
        self.clips = 0

    @torch.no_grad()
    def update(self, outputs, target_force, target_mask, mask_threshold: float = 0.5) -> None:
        pred_force, mask_logits = outputs["force"], outputs["mask_logits"]
        contact = target_mask > 0.5
        error = (pred_force - target_force) * contact
        self.abs_err += error.abs().sum(dim=(0, 1, 3, 4))
        self.sq_err += error.pow(2).sum(dim=(0, 1, 3, 4))
        self.contact += contact.sum(dim=(0, 1, 3, 4)).float()

        predicted = mask_logits.sigmoid() >= mask_threshold
        self.intersection += (predicted & contact).sum(dim=(0, 1, 3, 4)).float()
        self.union += (predicted | contact).sum(dim=(0, 1, 3, 4)).float()
        self.clips += pred_force.shape[0]

    def summarize(self) -> Dict:
        denom = self.contact.clamp_min(1.0)
        mae = self.abs_err / denom
        rmse = torch.sqrt(self.sq_err / denom)
        iou = self.intersection / self.union.clamp_min(1.0)
        seen = self.union > 0
        return {
            "clips": self.clips,
            "mae_by_channel": {n: float(v) for n, v in zip(CHANNELS, mae)},
            "rmse_by_channel": {n: float(v) for n, v in zip(CHANNELS, rmse)},
            "iou_by_channel": {n: float(v) for n, v in zip(CHANNELS, iou)},
            "mae": float(mae.mean()),
            "rmse": float(rmse.mean()),
            "miou": float(iou[seen].mean()) if bool(seen.any()) else float("nan"),
        }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate NeoForce: force MAE / RMSE and mask mIoU.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", default="configs/neoforce.json")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--split", default="splits/test.jsonl")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--max-batches", type=int, default=0, help="0 = the whole split.")
    parser.add_argument("--sample-stride", type=int, default=256,
                        help="Clips are drawn every N frames; the default keeps a full pass tractable.")
    parser.add_argument("--mask-threshold", type=float, default=0.5)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--sensor-device", default="cpu", help="Device for the tactile conversion model in the workers.")
    parser.add_argument("--output", default=None, help="Optional JSON path for the report.")
    args = parser.parse_args()

    device = torch.device(args.device)
    model, model_cfg, force_norm = load_model(args.checkpoint, device)
    print(f"force scale: {force_norm.describe() if force_norm else 'none recorded'}")

    raw = dict(json.loads(Path(args.config).read_text()).get("eval_data") or json.loads(Path(args.config).read_text())["data"])
    raw.update({
        "root": args.data_root,
        "index": args.split,
        "sensor_device": args.sensor_device,
        "sample_stride": args.sample_stride,
        "image_size": tuple(model_cfg.image_size),
        # Inputs must be scaled exactly as during training: a mismatch here corrupts the
        # model's input, not just the reported numbers.
        "force_mean": list(force_norm.mean) if force_norm else None,
        "force_std": list(force_norm.scale) if force_norm else None,
        "force_norm_mode": "none",
        "force_stats": None,
    })
    known = OpenNeoDataConfig.__dataclass_fields__
    dataset = OpenNeoDataset(OpenNeoDataConfig(**{k: v for k, v in raw.items() if k in known}))
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=device.type == "cuda")
    print(f"split {args.split}: {len(dataset.entries)} episode/wrist pairs -> {len(dataset)} clips")

    metrics = Metrics(device)
    total = min(args.max_batches, len(loader)) if args.max_batches > 0 else len(loader)
    for index, batch in enumerate(tqdm(loader, total=total, desc="evaluate")):
        if args.max_batches > 0 and index >= args.max_batches:
            break
        rgb = batch["rgb"].to(device, non_blocking=True)
        force = batch["force"].to(device, non_blocking=True)
        mask = batch["mask"].to(device, non_blocking=True)
        with torch.no_grad():
            outputs = model(rgb, force=force, mask=mask, touch_available=batch["touch_available"].to(device))
        metrics.update(outputs, force, mask, args.mask_threshold)

    report = metrics.summarize()
    print(f"\n{'channel':<10} {'MAE':>14} {'RMSE':>14} {'IoU':>8}")
    for name in CHANNELS:
        print(f"{name:<10} {report['mae_by_channel'][name]:>14.6e} "
              f"{report['rmse_by_channel'][name]:>14.6e} {report['iou_by_channel'][name]:>8.4f}")
    print(f"\n  MAE  : {report['mae']:.6e}")
    print(f"  RMSE : {report['rmse']:.6e}")
    print(f"  mIoU : {report['miou']:.4f}")
    print(f"  clips: {report['clips']}")

    if args.output:
        payload = {"checkpoint": args.checkpoint, "split": args.split,
                   "mask_threshold": args.mask_threshold, "metrics": report}
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(payload, indent=2))
        print(f"  wrote {args.output}")


if __name__ == "__main__":
    main()
