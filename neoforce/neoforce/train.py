"""NeoForce training: force-field reconstruction + JEPA latent prediction.

Single-node multi-GPU via ``torch.distributed.run``. Force fields are inferred online
from the raw tactile videos by the frozen tactile conversion model, so no cached tensors
are needed and the ground truth is exactly what that model produces.

    torchrun --nproc_per_node 8 -m neoforce.train \\
      --config configs/neoforce.json \\
      --data-root /path/to/OpenNeoData \\
      --output-dir outputs/neoforce
"""

import argparse
import json
import math
import os
import time
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from neoforce.data import ForceNormalizer, OpenNeoDataConfig, OpenNeoDataset, apply_force_norm_to_data_config
from neoforce.evaluate import Metrics
from neoforce.losses import DINOLoss, LossWeights, NeoForceLoss, add_jepa_distillation_losses, update_ema
from neoforce.models import NeoForceConfig, NeoForceJEPA, multiblock_masks


def load_json(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_model_config(raw: Dict) -> NeoForceConfig:
    known = NeoForceConfig.__dataclass_fields__
    return NeoForceConfig(**{k: v for k, v in raw.items() if k in known})


def build_data_config(raw: Dict, root: Optional[str] = None, split: Optional[str] = None) -> OpenNeoDataConfig:
    cfg = dict(raw)
    if root:
        cfg["root"] = root
    if split:
        cfg["index"] = split
    for key in ("image_size", "mask_thresholds"):
        if cfg.get(key) is not None:
            cfg[key] = tuple(cfg[key])
    known = OpenNeoDataConfig.__dataclass_fields__
    return OpenNeoDataConfig(**{k: v for k, v in cfg.items() if k in known})


def resolve_force_norm(data_cfg: OpenNeoDataConfig, stats_path: Optional[str], is_main: bool) -> ForceNormalizer:
    """Turn the recorded quantiles into the concrete scale the dataset applies."""
    normalizer = apply_force_norm_to_data_config(data_cfg, stats_path)
    if is_main:
        print(f"Force normalization: {normalizer.describe()}")
        if normalizer.shifts_zero:
            print(
                "WARNING: the normalization has a non-zero mean, so zero no longer means "
                "'no contact'. The touch-dropout and vision-only paths feed literal zeros "
                "and would be silently corrupted."
            )
    return normalizer


def mirror_force_norm(source: OpenNeoDataConfig, target: OpenNeoDataConfig) -> None:
    """Give eval the training scale.

    Eval must never derive its own scale from its own split, or its numbers would be in
    a different unit system than training while still looking comparable.
    """
    target.force_mean = list(source.force_mean) if source.force_mean is not None else None
    target.force_std = list(source.force_std) if source.force_std is not None else None
    target.force_stats = source.force_stats
    target.force_norm_stats = source.force_norm_stats
    target.force_norm_mode = "none"


def build_loader(data_cfg: OpenNeoDataConfig, train_cfg: Dict, device, shuffle: bool, distributed: bool) -> DataLoader:
    dataset = OpenNeoDataset(data_cfg)
    sampler = DistributedSampler(dataset, shuffle=shuffle, drop_last=shuffle) if distributed else None
    workers = int(train_cfg.get("num_workers", 8))
    kwargs = {}
    if workers > 0 and str(data_cfg.sensor_device).startswith("cuda"):
        # CUDA cannot be re-initialized in forked workers once the parent has touched it.
        kwargs = {"multiprocessing_context": "spawn", "persistent_workers": True}
    return DataLoader(
        dataset,
        batch_size=int(train_cfg.get("batch_size", 1)),
        shuffle=shuffle if sampler is None else False,
        sampler=sampler,
        num_workers=workers,
        pin_memory=device.type == "cuda",
        drop_last=shuffle,
        **kwargs,
    )


def setup_distributed(device_arg: Optional[str]):
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size <= 1:
        device = torch.device(device_arg or ("cuda" if torch.cuda.is_available() else "cpu"))
        return False, 0, 1, device
    if not torch.cuda.is_available():
        raise RuntimeError("Distributed training requires CUDA for the nccl backend.")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")
    return True, dist.get_rank(), world_size, torch.device("cuda", local_rank)


def param_groups(model, trunk_lr: float, head_lr: float, weight_decay: float):
    trunk, heads = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        (trunk if name.startswith(("blocks.", "vision_patch_embed.", "patch_pos_embed", "cls_token")) else heads).append(param)
    return [
        {"params": trunk, "lr": trunk_lr, "weight_decay": weight_decay},
        {"params": heads, "lr": head_lr, "weight_decay": weight_decay},
    ]


def apply_lr_warmup(optimizer, base_lrs, step: int, warmup_steps: int) -> None:
    if warmup_steps <= 0:
        return
    scale = min(1.0, float(step + 1) / float(warmup_steps))
    for group, base_lr in zip(optimizer.param_groups, base_lrs):
        group["lr"] = base_lr * scale


def autocast_dtype(train_cfg: Dict) -> torch.dtype:
    dtype = str(train_cfg.get("amp_dtype", "bf16")).lower()
    if dtype in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if dtype in {"fp16", "float16", "half"}:
        return torch.float16
    raise ValueError(f"Unsupported amp dtype `{dtype}`. Use `bf16` or `fp16`.")


class TrainStep(torch.nn.Module):
    """Encoder forward and JEPA predictor inside one ``forward()``.

    This exists for DDP correctness. DDP decides which parameters took part in a step by
    walking the autograd graph back from what ``forward()`` returns; a predictor invoked
    after the encoder call, outside the wrapped module, never has its gradients
    all-reduced, so every rank would drift to a different predictor while the trunk
    stayed in sync.
    """

    def __init__(self, encoder: NeoForceJEPA):
        super().__init__()
        self.encoder = encoder

    def forward(self, rgb, force, mask, touch_available, vision_patch_mask, touch_patch_mask, grid):
        outputs = self.encoder(
            rgb,
            force=force,
            mask=mask,
            touch_available=touch_available,
            vision_patch_mask=vision_patch_mask,
            touch_patch_mask=touch_patch_mask,
        )
        pred_v = self.encoder.jepa_predict(outputs["patch_v"], vision_patch_mask, grid, modality=0)
        pred_t = self.encoder.jepa_predict(outputs["patch_t"], touch_patch_mask, grid, modality=1)
        return outputs, pred_v, pred_t


def save_checkpoint(path: Path, encoder, teacher, optimizer, model_cfg, data_cfg, step: int, force_norm) -> None:
    torch.save(
        {
            "model": encoder.state_dict(),
            "teacher": teacher.state_dict() if teacher is not None else None,
            "optimizer": optimizer.state_dict(),
            "model_cfg": model_cfg.__dict__,
            "data_cfg": data_cfg.__dict__,
            # Stored at the top level as well as inside data_cfg: evaluation needs the
            # scale to invert the normalization before computing metrics.
            "force_norm": force_norm.to_dict() if force_norm is not None else None,
            "step": step,
        },
        path,
    )


@torch.no_grad()
def run_eval(encoder, loader, device, max_batches: int, mask_threshold: float = 0.5) -> Dict[str, float]:
    """Validation reports the three evaluation metrics, not the training losses.

    Loss values track the objective, which the training log already shows every step. A
    validation pass is for the quantities the model is judged on, computed exactly as
    ``neoforce.evaluate`` computes them on the test split so the two are comparable.
    """
    if loader is None:
        return {}
    was_training = encoder.training
    encoder.eval()
    metrics = Metrics(device)
    for index, batch in enumerate(loader):
        if max_batches > 0 and index >= max_batches:
            break
        rgb = batch["rgb"].to(device, non_blocking=True)
        force = batch["force"].to(device, non_blocking=True)
        mask = batch["mask"].to(device, non_blocking=True)
        outputs = encoder(rgb, force=force, mask=mask, touch_available=batch["touch_available"].to(device))
        metrics.update(outputs, force, mask, mask_threshold)
    if was_training:
        encoder.train()
    if metrics.clips == 0:
        return {}
    report = metrics.summarize()
    return {"eval/mae": report["mae"], "eval/rmse": report["rmse"], "eval/miou": report["miou"]}


def train(args: argparse.Namespace) -> None:
    distributed, rank, world_size, device = setup_distributed(args.device)
    is_main = rank == 0
    raw_cfg = load_json(args.config)
    model_cfg = build_model_config(raw_cfg.get("model", {}))

    data_raw = dict(raw_cfg.get("data", {}))
    data_raw["image_size"] = list(model_cfg.image_size)
    data_cfg = build_data_config(data_raw, args.data_root, args.train_split)
    force_norm = resolve_force_norm(data_cfg, args.force_stats, is_main)
    weights = LossWeights(**{k: v for k, v in raw_cfg.get("loss_weights", {}).items() if k in LossWeights.__dataclass_fields__})
    train_cfg = raw_cfg.get("train", {})

    loader = build_loader(data_cfg, train_cfg, device, shuffle=True, distributed=distributed)
    eval_loader = None
    if is_main and raw_cfg.get("eval_data"):
        eval_raw = dict(raw_cfg["eval_data"])
        eval_raw["image_size"] = list(model_cfg.image_size)
        eval_cfg = build_data_config(eval_raw, args.data_root, args.eval_split)
        mirror_force_norm(data_cfg, eval_cfg)
        eval_loader = build_loader(eval_cfg, train_cfg, device, shuffle=False, distributed=False)

    encoder = NeoForceJEPA(model_cfg).to(device)
    if args.dinov2_checkpoint:
        result = encoder.load_dinov2_weights(args.dinov2_checkpoint, strict=False)
        if is_main:
            print(f"Loaded DINOv2-compatible weights: {result}")

    step_module = TrainStep(encoder).to(device)
    if distributed:
        step_module = DDP(
            step_module,
            device_ids=[device.index],
            output_device=device.index,
            find_unused_parameters=bool(train_cfg.get("find_unused_parameters", True)),
        )

    objective = NeoForceLoss(weights).to(device)
    # Param groups come from the encoder, not the wrapper: the trunk/head split matches
    # on parameter names, and wrapping prefixes them, which would put the whole trunk on
    # the head learning rate.
    optimizer = torch.optim.AdamW(
        param_groups(
            encoder,
            trunk_lr=float(train_cfg.get("trunk_lr", 5e-6)),
            head_lr=float(train_cfg.get("head_lr", 8e-5)),
            weight_decay=float(train_cfg.get("weight_decay", 0.05)),
        )
    )
    base_lrs = [group["lr"] for group in optimizer.param_groups]
    warmup_steps = int(train_cfg.get("warmup_steps", 0))

    use_amp = bool(train_cfg.get("amp", True)) and device.type == "cuda"
    amp_dtype = autocast_dtype(train_cfg) if use_amp else torch.float32
    # GradScaler is for fp16 only; bf16 has the range that made scaling necessary.
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp and amp_dtype == torch.float16)

    out_dir = Path(args.output_dir)
    if is_main:
        out_dir.mkdir(parents=True, exist_ok=True)

    # The teacher is a frozen EMA copy; it is built by construction + load_state_dict
    # rather than deepcopy because the DINO head's weight_norm parameterization makes
    # its `weight` a non-leaf tensor that cannot be deepcopied.
    teacher = NeoForceJEPA(model_cfg).to(device)
    teacher.load_state_dict(encoder.state_dict())
    teacher.eval()
    for param in teacher.parameters():
        param.requires_grad_(False)
    dino_loss = DINOLoss(model_cfg.prototype_dim).to(device)

    max_steps = int(train_cfg.get("max_steps", 100000))
    epochs = int(math.ceil(max_steps / max(1, len(loader))))
    touch_dropout = float(train_cfg.get("touch_dropout", 0.0))
    teacher_temp = float(train_cfg.get("teacher_temp", 0.04))
    ema_base = float(train_cfg.get("ema_momentum", 0.992))
    log_every = int(train_cfg.get("log_every", 20))
    save_every = int(train_cfg.get("save_every", 5000))
    eval_every = int(train_cfg.get("eval_every", 0))
    eval_batches = int(train_cfg.get("eval_batches", 32))
    grad_clip = float(train_cfg.get("grad_clip", 1.0))

    global_step = 0
    started_at = time.perf_counter()
    step_module.train()
    try:
        for epoch in range(epochs):
            if distributed and hasattr(loader.sampler, "set_epoch"):
                loader.sampler.set_epoch(epoch)
            pbar = tqdm(loader, desc=f"epoch {epoch}", disable=not is_main)
            for batch in pbar:
                rgb = batch["rgb"].to(device, non_blocking=True)
                target_force = batch["force"].to(device, non_blocking=True)
                target_mask = batch["mask"].to(device, non_blocking=True)
                touch_available = batch["touch_available"].to(device, non_blocking=True)

                force_in, mask_in = target_force.clone(), target_mask.clone()
                if touch_dropout > 0.0:
                    drop = torch.rand(touch_available.shape, device=device) < touch_dropout
                    force_in = force_in.masked_fill(drop[:, :, None, None, None], 0.0)
                    mask_in = mask_in.masked_fill(drop[:, :, None, None, None], 0.0)
                    touch_available = touch_available & ~drop

                grid = (rgb.shape[-2] // model_cfg.patch_size, rgb.shape[-1] // model_cfg.patch_size)
                mask_kwargs = (rgb.shape[0], rgb.shape[1], grid, model_cfg.jepa_num_blocks,
                               model_cfg.jepa_block_scale, model_cfg.jepa_block_aspect, device)
                vision_patch_mask = multiblock_masks(*mask_kwargs)
                touch_patch_mask = multiblock_masks(*mask_kwargs)

                apply_lr_warmup(optimizer, base_lrs, global_step, warmup_steps)
                optimizer.zero_grad(set_to_none=True)
                with torch.cuda.amp.autocast(enabled=use_amp, dtype=amp_dtype):
                    outputs, pred_v, pred_t = step_module(
                        rgb, force_in, mask_in, touch_available, vision_patch_mask, touch_patch_mask, grid
                    )
                    losses = objective(outputs, target_force, target_mask)
                    with torch.no_grad():
                        teacher_outputs = teacher(
                            rgb, force=target_force, mask=target_mask,
                            touch_available=batch["touch_available"].to(device),
                        )
                    losses = add_jepa_distillation_losses(
                        losses, outputs, teacher_outputs, pred_v, pred_t, dino_loss,
                        weights, teacher_temp, vision_patch_mask, touch_patch_mask,
                    )

                scaler.scale(losses["loss"]).backward()
                scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(encoder.parameters(), grad_clip)
                scaler.step(optimizer)
                scaler.update()

                progress = min(1.0, global_step / max(1, max_steps))
                momentum = 1.0 - (1.0 - ema_base) * (math.cos(math.pi * progress) + 1.0) / 2.0
                update_ema(encoder, teacher, momentum)

                if is_main and global_step % log_every == 0:
                    pbar.set_postfix({k: f"{v.item():.4f}" for k, v in losses.items() if torch.is_tensor(v)})
                if is_main and eval_loader is not None and eval_every > 0 and global_step > 0 and global_step % eval_every == 0:
                    stats = run_eval(encoder, eval_loader, device, eval_batches)
                    if stats:
                        print(f"[step {global_step}] " + "  ".join(f"{k.split('/')[1]} {v:.4f}"
                                                                   for k, v in stats.items()), flush=True)
                if is_main and global_step > 0 and global_step % save_every == 0:
                    save_checkpoint(out_dir / f"neoforce_step_{global_step}.pt", encoder, teacher,
                                    optimizer, model_cfg, data_cfg, global_step, force_norm)

                global_step += 1
                if global_step >= max_steps:
                    if is_main:
                        save_checkpoint(out_dir / "neoforce_last.pt", encoder, teacher,
                                        optimizer, model_cfg, data_cfg, global_step, force_norm)
                        print(f"done: {global_step} steps in {(time.perf_counter() - started_at)/3600:.2f} h")
                    return
    finally:
        if distributed and dist.is_initialized():
            dist.destroy_process_group()


def main() -> None:
    parser = argparse.ArgumentParser(description="Train NeoForce on OpenNeoData.")
    parser.add_argument("--config", default="configs/neoforce.json")
    parser.add_argument("--data-root", required=True, help="OpenNeoData root directory.")
    parser.add_argument("--train-split", default=None, help="Overrides data.index in the config.")
    parser.add_argument("--eval-split", default=None, help="Overrides eval_data.index in the config.")
    parser.add_argument("--force-stats", default=None, help="Overrides data.force_stats in the config.")
    parser.add_argument("--output-dir", default="outputs/neoforce")
    parser.add_argument("--dinov2-checkpoint", default=None, help="Optional DINOv2 checkpoint to warm-start the trunk.")
    parser.add_argument("--device", default=None)
    train(parser.parse_args())


if __name__ == "__main__":
    main()
