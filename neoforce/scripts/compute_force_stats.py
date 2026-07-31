"""Compute per-channel force quantiles over the NeoForce training split.

Writes the JSON that ``data.force_stats`` points at. The scale it records is what keeps
the dense branch trainable: raw force values are ~1e-4 N, which a Huber loss with the
usual beta=1 treats as pure noise next to the O(1) distillation terms.

Clips are sampled through the same tactile conversion model training uses, and the contact
population is the engine's own mask, so the statistics describe exactly the pixels the
force loss is computed over.

    python scripts/compute_force_stats.py \
      --config configs/neoforce.json \
      --data-root /path/to/OpenNeoData \
      --split splits/train.jsonl \
      --samples 800 --device cuda \
      --output splits/force_stats_train.json
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from neoforce.data.force_norm import (  # noqa: E402
    CHANNELS,
    QUANTILE_LEVELS,
    ForceStats,
    quantile_key,
)


class Reservoir:
    """Fixed-memory uniform sample of a stream, for quantile estimation.

    Vectorized algorithm R. When several incoming items land on the same slot the
    last write wins; with the sample sizes used here that bias is far below the
    Monte-Carlo error of the quantiles themselves.
    """

    def __init__(self, capacity: int, rng: np.random.Generator):
        self.capacity = int(capacity)
        self.buffer = np.empty(self.capacity, dtype=np.float64)
        self.size = 0
        self.seen = 0
        self.rng = rng

    def add(self, values: np.ndarray) -> None:
        values = np.asarray(values, dtype=np.float64).ravel()
        if values.size == 0:
            return
        if self.size < self.capacity:
            take = min(self.capacity - self.size, values.size)
            self.buffer[self.size : self.size + take] = values[:take]
            self.size += take
            self.seen += take
            values = values[take:]
            if values.size == 0:
                return
        positions = self.seen + np.arange(1, values.size + 1)
        slots = self.rng.integers(0, positions)
        keep = slots < self.capacity
        if keep.any():
            self.buffer[slots[keep]] = values[keep]
        self.seen += values.size

    def quantiles(self, levels: Tuple[float, ...]) -> Dict[str, float]:
        if self.size == 0:
            return {quantile_key(level): 0.0 for level in levels}
        sample = self.buffer[: self.size]
        values = np.quantile(sample, levels)
        return {quantile_key(level): float(value) for level, value in zip(levels, values)}


class ChannelAccumulator:
    def __init__(self, capacity: int, rng: np.random.Generator):
        self.contact = Reservoir(capacity, rng)
        self.all = Reservoir(capacity, rng)
        self.contact_abs_sum = 0.0
        self.contact_abs_sq_sum = 0.0
        self.contact_signed_sum = 0.0
        self.contact_count = 0
        self.all_abs_sum = 0.0
        self.all_abs_sq_sum = 0.0
        self.all_signed_sum = 0.0
        self.all_count = 0


def accumulate(
    accumulators: List[ChannelAccumulator],
    force: np.ndarray,
    thresholds: np.ndarray,
    all_pixels_per_frame: int,
    rng: np.random.Generator,
    contact_mask: Optional[np.ndarray] = None,
) -> None:
    """``force`` is ``(T, 6, H, W)`` in raw physical units.

    ``contact_mask`` (same shape, boolean) overrides threshold-based contact detection.
    The online source passes the dataset's own derived mask so the sampled population
    is exactly the one the force loss is computed over.
    """
    frames, chans = force.shape[0], force.shape[1]
    flat = force.reshape(frames, chans, -1)
    flat_mask = contact_mask.reshape(frames, chans, -1) if contact_mask is not None else None
    for channel in range(chans):
        acc = accumulators[channel]
        values = flat[:, channel, :].ravel()
        magnitude = np.abs(values)

        if flat_mask is not None:
            contact = flat_mask[:, channel, :].ravel().astype(bool)
        else:
            contact = magnitude >= thresholds[channel]
        contact_values = values[contact]
        contact_magnitude = magnitude[contact]
        acc.contact.add(contact_magnitude)
        acc.contact_abs_sum += float(contact_magnitude.sum())
        acc.contact_abs_sq_sum += float((contact_magnitude**2).sum())
        acc.contact_signed_sum += float(contact_values.sum())
        acc.contact_count += int(contact_magnitude.size)

        acc.all_abs_sum += float(magnitude.sum())
        acc.all_abs_sq_sum += float((magnitude**2).sum())
        acc.all_signed_sum += float(values.sum())
        acc.all_count += int(values.size)
        if all_pixels_per_frame > 0 and values.size > 0:
            take = min(all_pixels_per_frame * frames, values.size)
            picks = rng.integers(0, values.size, size=take)
            acc.all.add(magnitude[picks])


def scan_openneo(
    args: argparse.Namespace,
    data_raw: Dict,
    accumulators: List[ChannelAccumulator],
    per_channel_thresholds: np.ndarray,
    rng: np.random.Generator,
) -> Tuple[int, int, Tuple[int, int]]:
    """Sample force fields through OpenNeoDataset, using the tactile conversion model.

    Uses the dataset's own contact mask, so the sampled population is exactly the one
    the force loss sees.
    """
    from neoforce.data import OpenNeoDataConfig, OpenNeoDataset

    cfg_raw = dict(data_raw)
    cfg_raw.update(
        {
            "root": args.data_root,
            # Statistics must be gathered in raw physical units.
            "force_mean": None,
            "force_std": None,
            "force_norm_mode": "none",
            "force_stats": None,
            "sensor_device": args.device,
        }
    )
    cfg_raw["index"] = args.split
    for key in ("image_size", "mask_thresholds"):
        if cfg_raw.get(key) is not None:
            cfg_raw[key] = tuple(cfg_raw[key])
    known = OpenNeoDataConfig.__dataclass_fields__
    dataset = OpenNeoDataset(OpenNeoDataConfig(**{k: v for k, v in cfg_raw.items() if k in known}))

    total = len(dataset)
    count = min(args.samples, total)
    indices = rng.choice(total, size=count, replace=False)
    print(f"sampling {count} of {total} OpenNeoData sequences (conversion model on {args.device})")

    frames = 0
    resolution = (0, 0)
    for step, index in enumerate(indices):
        try:
            sample = dataset[int(index)]
        except Exception as exc:  # noqa: BLE001 - a bad episode must not abort the scan
            print(f"  [skip] sequence {int(index)}: {exc}")
            continue
        force = sample["force"].numpy()
        mask = sample["mask"].numpy() > 0.5
        resolution = (int(force.shape[-2]), int(force.shape[-1]))
        accumulate(accumulators, force, per_channel_thresholds, args.all_pixels_per_frame, rng, contact_mask=mask)
        frames += force.shape[0]
        if (step + 1) % 20 == 0:
            print(f"  {step + 1}/{count} sequences, {frames} frames")
    return len(dataset.entries), frames, resolution


def build_stats(
    accumulators: List[ChannelAccumulator],
    args: argparse.Namespace,
    thresholds: Tuple[float, ...],
    resolution: Tuple[int, int],
    num_episodes: int,
    num_frames: int,
) -> ForceStats:
    abs_quantiles: Dict[str, Dict[str, Dict[str, float]]] = {"contact": {}, "all": {}}
    abs_mean: Dict[str, Dict[str, float]] = {"contact": {}, "all": {}}
    abs_std: Dict[str, Dict[str, float]] = {"contact": {}, "all": {}}
    signed_mean: Dict[str, Dict[str, float]] = {"contact": {}, "all": {}}
    contact_ratio: Dict[str, float] = {}
    sampled: Dict[str, int] = {}

    for name, acc in zip(CHANNELS, accumulators):
        abs_quantiles["contact"][name] = acc.contact.quantiles(QUANTILE_LEVELS)
        abs_quantiles["all"][name] = acc.all.quantiles(QUANTILE_LEVELS)

        contact_n = max(acc.contact_count, 1)
        all_n = max(acc.all_count, 1)
        contact_mean = acc.contact_abs_sum / contact_n
        all_mean = acc.all_abs_sum / all_n
        abs_mean["contact"][name] = contact_mean
        abs_mean["all"][name] = all_mean
        abs_std["contact"][name] = float(np.sqrt(max(0.0, acc.contact_abs_sq_sum / contact_n - contact_mean**2)))
        abs_std["all"][name] = float(np.sqrt(max(0.0, acc.all_abs_sq_sum / all_n - all_mean**2)))
        signed_mean["contact"][name] = acc.contact_signed_sum / contact_n
        signed_mean["all"][name] = acc.all_signed_sum / all_n
        contact_ratio[name] = acc.contact_count / all_n
        sampled[name] = acc.contact.size

    return ForceStats(
        channels=CHANNELS,
        quantile_levels=QUANTILE_LEVELS,
        abs_quantiles=abs_quantiles,
        abs_mean=abs_mean,
        abs_std=abs_std,
        signed_mean=signed_mean,
        contact_ratio=contact_ratio,
        mask_thresholds=thresholds,
        resolution=resolution,
        num_episodes=num_episodes,
        num_frames=num_frames,
        num_sampled_values=sampled,
        h5_filename=args.h5_filename,
        data_root=str(args.data_root),
        split=Path(args.split).stem,
    )


def write_stats(
    args: argparse.Namespace,
    accumulators: List[ChannelAccumulator],
    thresholds: Tuple[float, ...],
    resolution: Tuple[int, int],
    num_episodes: int,
    total_frames: int,
) -> None:
    stats = build_stats(accumulators, args, thresholds, resolution, num_episodes, total_frames)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8") as f:
        json.dump(stats.to_dict(), f, indent=2)

    print(f"\nwrote {output}  ({total_frames} frames, resolution {resolution})")
    print(f"{'channel':<10} {'contact%':>9} {'q50':>12} {'q90':>12} {'q99':>12} {'q99.9':>12} {'max':>12}")
    for name in CHANNELS:
        q = stats.abs_quantiles["contact"][name]
        print(
            f"{name:<10} {100 * stats.contact_ratio[name]:>8.3f}% "
            f"{q['q50']:>12.4g} {q['q90']:>12.4g} {q['q99']:>12.4g} {q['q99.9']:>12.4g} {q['q100']:>12.4g}"
        )
    print("\nScale that force_norm_quantile=0.99 would select:")
    print("  " + ", ".join(f"{name}={value:.6g}" for name, value in zip(CHANNELS, stats.scale_at(0.99))))


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute per-channel tactile force quantiles for A1 normalization.")
    parser.add_argument("--config", default=None, help="Training config; supplies h5_filename / mask_thresholds / wrist_sides.")
    parser.add_argument("--split", required=True, help="OpenNeoData train.jsonl to gather statistics over.")
    parser.add_argument("--samples", type=int, default=800, help="number of clips to sample.")
    parser.add_argument("--device", default="cpu", help="device for the tactile conversion model.")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--mask-thresholds", nargs=3, type=float, default=None, help="fx fy fz, applied to raw maps.")
    parser.add_argument("--max-samples", type=int, default=2_000_000, help="Reservoir capacity per channel per population.")
    parser.add_argument("--all-pixels-per-frame", type=int, default=512)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    data_raw: Dict = {}
    if args.config:
        with open(args.config, "r", encoding="utf-8") as f:
            data_raw = json.load(f).get("data", {})
    thresholds = tuple(args.mask_thresholds or data_raw.get("mask_thresholds", (3.6e-05, 2.0e-05, 6.2500003e-05)))

    rng = np.random.default_rng(args.seed)
    per_channel_thresholds = np.array(list(thresholds) * 2, dtype=np.float64)
    accumulators = [ChannelAccumulator(args.max_samples, rng) for _ in CHANNELS]

    num_episodes, total_frames, resolution = scan_openneo(
        args, data_raw, accumulators, per_channel_thresholds, rng
    )
    if total_frames == 0:
        raise SystemExit("No sequences could be decoded; check --data-root and --split.")
    write_stats(args, accumulators, thresholds, resolution, num_episodes, total_frames)



if __name__ == "__main__":
    main()
