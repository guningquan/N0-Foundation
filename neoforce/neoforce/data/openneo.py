"""OpenNeoData dataset with online force-field ground truth.

OpenNeoData is LeRobot-format, which differs from the episode-directory layout the
a per-episode directory layout would in one way that matters: **many episodes share a
single mp4**, and an episode is the ``[from_timestamp, to_timestamp]`` slice of it.
Seeking is therefore by timestamp, not by frame index from 0.

    <root>/<robot>/meta/info.json                       feature list, fps
    <root>/<robot>/meta/episodes/chunk-*/file-*.parquet  episode index
    <root>/<robot>/videos/<key>/chunk-XXX/file-XXX.mp4   video streams

Streams used per sample, for one wrist side:

    observation.images.<side>_wrist_view          RGB input
    observation.images.<side>_wrist_left_tactile  -> force channels 0..2
    observation.images.<side>_wrist_right_tactile -> force channels 3..5

Force fields are produced by the tactile conversion model at ``(360, 640)`` and used
directly as training GT, along with the pipeline's own contact mask. No H5 cache is
read or written.

Scanning 228k episodes of parquet takes minutes, so the sequence index is built once
by ``scripts/build_openneo_index.py`` and loaded from JSON here.
"""

from __future__ import annotations

import bisect
import json
import math
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from .force_norm import normalizer_from_data_config

TACTILE_SIDES = ("left", "right")
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def rgb_key(wrist_side: str) -> str:
    return f"observation.images.{wrist_side}_wrist_view"


def tactile_key(wrist_side: str, sensor_side: str) -> str:
    return f"observation.images.{wrist_side}_wrist_{sensor_side}_tactile"


def category_id(task: str, robot: str) -> str:
    """``<task_slug>-<robot>``: the category identifier used by the split files."""
    return f"{'_'.join(str(task).lower().split())}-{robot}"


@dataclass
class OpenNeoDataConfig:
    """Config for :class:`OpenNeoDataset`.

    Carries the force-normalization fields so the same
    plan-A1 machinery (``scripts/compute_force_stats.py`` -> ``force_std`` ->
    checkpoint -> de-normalized eval) applies unchanged.
    """

    root: str = ""
    index: str = ""
    seq_len: int = 4
    sample_stride: int = 1
    image_size: Tuple[int, int] = (360, 640)
    force_chans: int = 6
    mask_chans: int = 6
    fps: float = 30.0
    normalize_rgb: bool = True

    # Subset selection. `categories` is an explicit allow-list of `<task_slug>-<robot>`
    # ids; `categories_json` reads one from a file (a list, or {"categories": [...]},
    # or the keys of a mapping keyed by category id).
    categories: Optional[Sequence[str]] = None
    categories_json: Optional[str] = None
    robots: Optional[Sequence[str]] = None
    max_episodes: Optional[int] = None
    max_sequences_per_episode: int = 0

    # Per-channel force normalization; see data/force_norm.py.
    force_mean: Optional[Sequence[float]] = None
    force_std: Optional[Sequence[float]] = None
    force_stats: Optional[str] = None
    force_norm_mode: str = "none"
    force_norm_quantile: float = 0.99
    force_norm_population: str = "contact"
    force_norm_stats: Optional[Dict[str, Any]] = None

    # Tactile conversion model.
    sensor_code_dir: Optional[str] = None
    sensor_device: str = "cpu"
    sensor_gpu_batch: int = 16
    use_engine_mask: bool = True
    mask_thresholds: Tuple[float, float, float] = (3.6e-05, 2.0e-05, 6.2500003e-05)

    retries: int = 3


def entries_from_split_jsonl(path: str, root: Path) -> List[Dict]:
    """Expand a ``train.jsonl``/``test.jsonl`` split into per-wrist-side entries.

    Each line is one episode carrying a ``videos`` map of stream -> {path, from_timestamp,
    to_timestamp}, with paths relative to the dataset root. A wrist side is emitted only
    when its RGB view and both tactile streams are present; a side missing any of them is
    skipped rather than filled in, because a fabricated stream would become fabricated GT.
    """
    entries: List[Dict] = []
    skipped_sides = 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            videos = record.get("videos") or {}
            for side in ("left", "right"):
                needed = [rgb_key(side), tactile_key(side, "left"), tactile_key(side, "right")]
                if not all(key in videos for key in needed):
                    skipped_sides += 1
                    continue
                streams = {
                    label: {
                        "path": str(root / videos[key]["path"]),
                        "t0": float(videos[key]["from_timestamp"]),
                        "t1": float(videos[key]["to_timestamp"]),
                    }
                    for label, key in zip(("rgb", "left", "right"), needed)
                }
                entries.append(
                    {
                        "robot": record.get("robot", ""),
                        "task": record.get("task", ""),
                        "category": record.get("category", ""),
                        "episode_index": int(record.get("episode_index", -1)),
                        "length": int(record["length"]),
                        "wrist_side": side,
                        "rgb": streams["rgb"],
                        "tactile": {"left": streams["left"], "right": streams["right"]},
                    }
                )
    if skipped_sides:
        print(f"openneo split {Path(path).name}: skipped {skipped_sides} wrist sides with missing streams")
    return entries


def _load_category_filter(cfg: OpenNeoDataConfig) -> Optional[set]:
    allowed = set(cfg.categories) if cfg.categories else set()
    if cfg.categories_json:
        with open(cfg.categories_json, "r", encoding="utf-8") as f:
            raw = json.load(f)
        if isinstance(raw, dict):
            raw = raw.get("categories", list(raw.keys()))
        allowed.update(str(item["id"] if isinstance(item, dict) else item) for item in raw)
    return allowed or None


class OpenNeoDataset(Dataset):
    """Yields ``{rgb, force, mask, touch_available}``, same contract as the H5 datasets.

    ``force``/``mask`` are ``(K, 6, 360, 640)`` with channel order
    ``[left_fx, left_fy, left_fz, right_fx, right_fy, right_fz]``; ``force`` is
    normalized by the configured A1 scale, ``mask`` is binary.
    """

    def __init__(self, cfg: OpenNeoDataConfig):
        self.cfg = cfg
        self._engine = None
        self._reference_cache: Dict[Tuple[str, float], np.ndarray] = {}
        self._reference_cache_limit = 64

        if not cfg.index:
            raise ValueError(
                "OpenNeoDataset requires cfg.index: either a train.jsonl/test.jsonl split, "
                "or a JSON built by scripts/build_index.py."
            )
        if str(cfg.index).endswith(".jsonl"):
            # OpenNeoData split files carry relative stream paths, so a root is mandatory.
            if not cfg.root:
                raise ValueError("A .jsonl split needs cfg.root (the OpenNeoData root) to resolve video paths.")
            self.root = Path(cfg.root)
            entries = entries_from_split_jsonl(cfg.index, self.root)
        else:
            with open(cfg.index, "r", encoding="utf-8") as f:
                payload = json.load(f)
            entries = payload["sequences"] if isinstance(payload, dict) else payload
            self.root = Path(cfg.root or (payload.get("root", "") if isinstance(payload, dict) else ""))
            if isinstance(payload, dict) and payload.get("fps"):
                self.cfg.fps = float(payload["fps"])

        self.entries = self._filter_entries(entries)
        if not self.entries:
            raise ValueError(f"No episodes left after filtering the index {cfg.index}.")

        self.sample_stride = max(1, int(cfg.sample_stride))
        self.cumulative_lengths = self._build_lengths()

        self.force_normalizer, self.force_mean, self.force_std = self._force_norm_tensors()
        self.rgb_mean = torch.tensor(IMAGENET_MEAN, dtype=torch.float32).view(3, 1, 1)
        self.rgb_std = torch.tensor(IMAGENET_STD, dtype=torch.float32).view(3, 1, 1)

    # ------------------------------------------------------------------ index

    def _filter_entries(self, entries: List[Dict]) -> List[Dict]:
        allowed = _load_category_filter(self.cfg)
        robots = set(self.cfg.robots) if self.cfg.robots else None
        kept = []
        for entry in entries:
            if robots is not None and entry.get("robot") not in robots:
                continue
            if allowed is not None and entry.get("category") not in allowed:
                continue
            kept.append(entry)
        if self.cfg.max_episodes:
            kept = kept[: int(self.cfg.max_episodes)]
        return kept

    def _build_lengths(self) -> List[int]:
        total = 0
        cumulative = []
        for entry in self.entries:
            usable = max(0, int(entry["length"]) - self.cfg.seq_len + 1)
            count = (usable + self.sample_stride - 1) // self.sample_stride
            if self.cfg.max_sequences_per_episode > 0:
                count = min(count, int(self.cfg.max_sequences_per_episode))
            total += count
            cumulative.append(total)
        if total == 0:
            raise ValueError("Every episode is shorter than seq_len; nothing to sample.")
        return cumulative

    def _force_norm_tensors(self):
        normalizer = normalizer_from_data_config(self.cfg)
        mean, std = normalizer.as_tensors(self.cfg.force_chans)
        return normalizer, mean, std

    def __len__(self) -> int:
        return self.cumulative_lengths[-1]

    # ------------------------------------------------------------------ engine

    def _infer_engine(self):
        """Built lazily so the dataset stays picklable across DataLoader workers."""
        if self._engine is None:
            from ..visuo_tactile_conversion.conversion import DEFAULT_CODE_DIR, TactileConversionModel

            self._engine = TactileConversionModel(
                code_dir=self.cfg.sensor_code_dir or DEFAULT_CODE_DIR,
                device=self.cfg.sensor_device,
                gpu_batch=self.cfg.sensor_gpu_batch,
            )
        return self._engine

    # ------------------------------------------------------------------ video io

    def _read_clip(self, path: str, start_time: float, count: int) -> List[np.ndarray]:
        """Read ``count`` consecutive frames starting at ``start_time`` seconds.

        One seek, then sequential reads: the frames of a clip are adjacent at fps, so
        seeking per frame would cost a keyframe search each time. A short read (dropped
        or corrupt frames near the end of a stream) is padded with the last good frame.
        """
        import cv2

        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            raise OSError(f"failed to open video: {path}")
        try:
            cap.set(cv2.CAP_PROP_POS_MSEC, start_time * 1000.0)
            frames: List[np.ndarray] = []
            last = None
            for _ in range(count):
                ok, frame = cap.read()
                if not ok or frame is None:
                    if last is None:
                        raise OSError(f"no readable frame at t={start_time:.3f}s in {path}")
                    frame = last
                else:
                    last = frame
                frames.append(frame)
            return frames
        finally:
            cap.release()

    def _read_frame_at(self, path: str, timestamp: float) -> np.ndarray:
        import cv2

        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            raise OSError(f"failed to open video: {path}")
        try:
            cap.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000.0)
            ok, frame = cap.read()
            if not ok or frame is None:
                raise OSError(f"no readable frame at t={timestamp:.3f}s in {path}")
            return frame
        finally:
            cap.release()

    def _reference_gray(self, path: str, t0: float) -> np.ndarray:
        """Corrected grayscale flow reference (the episode's first tactile frame).

        Cached per (video, episode start): every sequence drawn from an episode shares
        it, and recomputing it would double the decode cost of every sample.
        """
        key = (str(path), round(float(t0), 6))
        cached = self._reference_cache.get(key)
        if cached is not None:
            return cached
        reference = self._infer_engine().reference_gray(self._read_frame_at(path, t0))
        if len(self._reference_cache) >= self._reference_cache_limit:
            self._reference_cache.pop(next(iter(self._reference_cache)))
        self._reference_cache[key] = reference
        return reference

    def _load_rgb(self, path: str, start_time: float, count: int) -> torch.Tensor:
        import cv2

        frames = self._read_clip(path, start_time, count)
        tensors = []
        for frame in frames:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            tensor = torch.from_numpy(np.transpose(rgb.astype(np.float32), (2, 0, 1))) / 255.0
            if tuple(tensor.shape[-2:]) != tuple(self.cfg.image_size):
                tensor = F.interpolate(
                    tensor.unsqueeze(0), size=tuple(self.cfg.image_size), mode="bilinear", align_corners=False
                ).squeeze(0)
            if self.cfg.normalize_rgb:
                tensor = (tensor - self.rgb_mean) / self.rgb_std
            tensors.append(tensor)
        return torch.stack(tensors, dim=0)

    def _load_force(self, entry: Dict, start_time: float) -> Tuple[torch.Tensor, torch.Tensor]:
        engine = self._infer_engine()
        forces, masks = [], []
        for sensor_side in TACTILE_SIDES:
            stream = entry["tactile"][sensor_side]
            reference = self._reference_gray(stream["path"], float(stream["t0"]))
            frames = self._read_clip(stream["path"], start_time, self.cfg.seq_len)
            force, mask = engine.force_maps(frames, reference)
            forces.append(force)
            masks.append(mask)
        force = torch.from_numpy(np.concatenate(forces, axis=1)).float()
        mask = torch.from_numpy(np.concatenate(masks, axis=1)).float()
        return force, mask

    def _derive_mask(self, force: torch.Tensor) -> torch.Tensor:
        thresholds = torch.tensor(
            list(self.cfg.mask_thresholds) * 2, dtype=force.dtype
        ).view(1, self.cfg.mask_chans, 1, 1)
        return (force.abs() >= thresholds).float()

    # ------------------------------------------------------------------ sampling

    def locate(self, idx: int) -> Tuple[Dict, int]:
        entry_idx = bisect.bisect_right(self.cumulative_lengths, idx)
        previous = 0 if entry_idx == 0 else self.cumulative_lengths[entry_idx - 1]
        frame_offset = (idx - previous) * self.sample_stride
        return self.entries[entry_idx], frame_offset

    def _load_sample(self, idx: int) -> Dict[str, torch.Tensor]:
        entry, frame_offset = self.locate(idx)
        fps = float(self.cfg.fps) or 30.0
        # Streams share one episode-relative clock; each stream carries its own t0
        # because episodes are slices of per-stream mp4s that need not be aligned.
        rgb_start = float(entry["rgb"]["t0"]) + frame_offset / fps
        rgb = self._load_rgb(entry["rgb"]["path"], rgb_start, self.cfg.seq_len)

        tactile_start = float(entry["tactile"]["left"]["t0"]) + frame_offset / fps
        force, mask = self._load_force(entry, tactile_start)
        if not self.cfg.use_engine_mask:
            mask = self._derive_mask(force)

        if force.shape[0] != rgb.shape[0]:
            raise RuntimeError(
                f"clip length mismatch: rgb={rgb.shape[0]} force={force.shape[0]} "
                f"(episode {entry.get('robot')}#{entry.get('episode_index')})"
            )

        force = (force - self.force_mean) / self.force_std
        return {
            "rgb": rgb,
            "force": force,
            "mask": mask,
            "touch_available": torch.ones(rgb.shape[0], dtype=torch.bool),
        }

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        # Corrupt/dropped frames are common at this scale; a single bad episode must
        # never abort a multi-day run. Retry, then fall back to another random sample.
        last_error: Optional[Exception] = None
        for attempt in range(max(1, self.cfg.retries)):
            try:
                return self._load_sample(idx)
            except Exception as exc:  # noqa: BLE001 - intentionally broad, see above
                last_error = exc
                idx = random.randrange(len(self)) if attempt else idx
        raise RuntimeError(f"failed to load a usable sample after {self.cfg.retries} retries: {last_error}")
