"""Per-channel tactile force normalization .

Raw force maps live at ~1e-5 and are fed straight into ``smooth_l1_loss`` with the
default ``beta=1.0``, i.e. deep in the quadratic region where ``dL/dpred = err``.
That leaves the dense branch with gradients some five orders of magnitude below the
distillation branches, so the force/mask heads never move away from their
near-zero initialization. Rescaling the target to O(1) is what makes the dense
loss competitive again.

Two invariants the normalization must preserve, both load-bearing:

1. **Zero maps to zero.** Non-contact pixels are exact zeros (``compute_pixel_force_map``
   hard-zeros them), the touch-dropout path fills dropped frames with zeros, and the
   vision-only inference path feeds an all-zero tactile tensor. A non-zero mean shift
   would silently turn "no force" into "mean force". Normalization is therefore
   **scale-only** and ``mean`` stays at 0 unless a caller explicitly overrides it.
2. **Sign is preserved.** ``fx``/``fy`` are signed shear components and
   ``force_direction_loss`` depends on their sign, so the scale is strictly positive
   and applied per channel, never per pixel.

Contact masks are derived from the *raw* maps before normalization (``_derive_mask``),
so ``mask_thresholds`` stay in physical units and are unaffected by anything here.
"""

from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, Tuple

import torch


CHANNELS: Tuple[str, ...] = (
    "left_fx",
    "left_fy",
    "left_fz",
    "right_fx",
    "right_fy",
    "right_fz",
)

# Recorded for every channel and both populations. The scale actually used for
# training is one of these, selected by ``force_norm_quantile``; the rest are kept
# in the checkpoint so a scale choice can be revisited without rescanning the data.
QUANTILE_LEVELS: Tuple[float, ...] = (
    0.5,
    0.75,
    0.9,
    0.95,
    0.99,
    0.995,
    0.999,
    0.9999,
    1.0,
)

POPULATIONS: Tuple[str, ...] = ("contact", "all")

_EPS = 1e-12


def quantile_key(level: float) -> str:
    """Stable dict key for a quantile level, in percent: ``0.99 -> 'q99'``,
    ``0.999 -> 'q99.9'``, ``1.0 -> 'q100'``."""
    return f"q{round(float(level) * 100.0, 6):g}"


@dataclass
class ForceStats:
    """Per-channel statistics of ``|force|`` over a dataset split.

    ``abs_quantiles[population][channel_name]`` maps a quantile key to a value.
    ``contact`` is the population of pixels passing ``mask_thresholds``; ``all``
    is every pixel including the zero background.
    """

    channels: Tuple[str, ...] = CHANNELS
    quantile_levels: Tuple[float, ...] = QUANTILE_LEVELS
    abs_quantiles: Dict[str, Dict[str, Dict[str, float]]] = field(default_factory=dict)
    abs_mean: Dict[str, Dict[str, float]] = field(default_factory=dict)
    abs_std: Dict[str, Dict[str, float]] = field(default_factory=dict)
    signed_mean: Dict[str, Dict[str, float]] = field(default_factory=dict)
    contact_ratio: Dict[str, float] = field(default_factory=dict)
    mask_thresholds: Tuple[float, ...] = ()
    resolution: Tuple[int, int] = (0, 0)
    num_episodes: int = 0
    num_frames: int = 0
    num_sampled_values: Dict[str, int] = field(default_factory=dict)
    h5_filename: str = ""
    data_root: str = ""
    split: str = ""

    def to_dict(self) -> Dict:
        return {
            "channels": list(self.channels),
            "quantile_levels": list(self.quantile_levels),
            "abs_quantiles": self.abs_quantiles,
            "abs_mean": self.abs_mean,
            "abs_std": self.abs_std,
            "signed_mean": self.signed_mean,
            "contact_ratio": self.contact_ratio,
            "mask_thresholds": list(self.mask_thresholds),
            "resolution": list(self.resolution),
            "num_episodes": self.num_episodes,
            "num_frames": self.num_frames,
            "num_sampled_values": self.num_sampled_values,
            "h5_filename": self.h5_filename,
            "data_root": self.data_root,
            "split": self.split,
        }

    @classmethod
    def from_dict(cls, raw: Dict) -> "ForceStats":
        return cls(
            channels=tuple(raw.get("channels", CHANNELS)),
            quantile_levels=tuple(raw.get("quantile_levels", QUANTILE_LEVELS)),
            abs_quantiles=raw.get("abs_quantiles", {}),
            abs_mean=raw.get("abs_mean", {}),
            abs_std=raw.get("abs_std", {}),
            signed_mean=raw.get("signed_mean", {}),
            contact_ratio=raw.get("contact_ratio", {}),
            mask_thresholds=tuple(raw.get("mask_thresholds", ())),
            resolution=tuple(raw.get("resolution", (0, 0))),
            num_episodes=int(raw.get("num_episodes", 0)),
            num_frames=int(raw.get("num_frames", 0)),
            num_sampled_values=raw.get("num_sampled_values", {}),
            h5_filename=raw.get("h5_filename", ""),
            data_root=raw.get("data_root", ""),
            split=raw.get("split", ""),
        )

    def scale_at(self, quantile: float, population: str = "contact") -> Tuple[float, ...]:
        """Per-channel scale = the requested quantile of ``|force|``.

        Raises rather than silently falling back: a wrong scale is exactly the
        failure this whole change exists to prevent, so it must not pass quietly.
        """
        if population not in self.abs_quantiles:
            raise KeyError(
                f"Force stats have no `{population}` population "
                f"(available: {sorted(self.abs_quantiles)})."
            )
        key = quantile_key(quantile)
        table = self.abs_quantiles[population]
        scale = []
        for name in self.channels:
            if name not in table:
                raise KeyError(f"Force stats are missing channel `{name}` in population `{population}`.")
            if key not in table[name]:
                raise KeyError(
                    f"Force stats have no quantile `{key}` for channel `{name}` "
                    f"(available: {sorted(table[name])})."
                )
            value = float(table[name][key])
            if not (value > 0.0):
                raise ValueError(
                    f"Force stats give a non-positive scale ({value}) for channel `{name}` "
                    f"at {key}/{population}; the channel is likely all-zero in this split."
                )
            scale.append(value)
        return tuple(scale)


@dataclass
class ForceNormalizer:
    """Scale-only affine map between physical force units and training units.

    ``encode`` is applied by the dataset; ``decode`` is applied at evaluation time
    so reported errors are back in physical units.
    """

    scale: Tuple[float, ...]
    mean: Tuple[float, ...] = ()
    quantile: Optional[float] = None
    population: str = "contact"
    source: str = ""

    def __post_init__(self) -> None:
        self.scale = tuple(float(value) for value in self.scale)
        if not self.scale:
            raise ValueError("ForceNormalizer requires a non-empty scale.")
        if any(value <= 0.0 for value in self.scale):
            raise ValueError(f"ForceNormalizer scale must be strictly positive, got {self.scale}.")
        self.mean = tuple(float(value) for value in self.mean) if self.mean else (0.0,) * len(self.scale)
        if len(self.mean) != len(self.scale):
            raise ValueError(
                f"ForceNormalizer mean/scale length mismatch: {len(self.mean)} vs {len(self.scale)}."
            )

    @classmethod
    def identity(cls, force_chans: int = len(CHANNELS)) -> "ForceNormalizer":
        return cls(scale=(1.0,) * force_chans, source="identity")

    @classmethod
    def from_stats(
        cls,
        stats: ForceStats,
        quantile: float = 0.99,
        population: str = "contact",
        source: str = "",
    ) -> "ForceNormalizer":
        return cls(
            scale=stats.scale_at(quantile, population),
            quantile=float(quantile),
            population=population,
            source=source,
        )

    @property
    def is_identity(self) -> bool:
        return all(value == 1.0 for value in self.scale) and all(value == 0.0 for value in self.mean)

    @property
    def shifts_zero(self) -> bool:
        """True when 0 does not map to 0; see the module docstring for why that matters."""
        return any(value != 0.0 for value in self.mean)

    def as_tensors(
        self,
        force_chans: int,
        dtype: torch.dtype = torch.float32,
        device: Optional[torch.device] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """``(mean, scale)`` shaped ``(C, 1, 1)`` for broadcasting over maps."""
        if len(self.scale) != force_chans:
            raise ValueError(
                f"ForceNormalizer has {len(self.scale)} channels but force_chans={force_chans}."
            )
        mean = torch.tensor(self.mean, dtype=dtype, device=device).view(force_chans, 1, 1)
        scale = torch.tensor(self.scale, dtype=dtype, device=device).view(force_chans, 1, 1).clamp_min(_EPS)
        return mean, scale

    def _broadcast(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if x.ndim < 3:
            raise ValueError(f"Expected a tensor with a channel dim at -3, got shape {tuple(x.shape)}.")
        chans = x.shape[-3]
        mean, scale = self.as_tensors(chans, dtype=x.dtype, device=x.device)
        return mean, scale

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Physical units -> training units. ``x`` is ``(..., C, H, W)``."""
        mean, scale = self._broadcast(x)
        return (x - mean) / scale

    def decode(self, x: torch.Tensor) -> torch.Tensor:
        """Training units -> physical units. ``x`` is ``(..., C, H, W)``."""
        mean, scale = self._broadcast(x)
        return x * scale + mean

    def decode_global(self, x: torch.Tensor) -> torch.Tensor:
        """Same as ``decode`` for the per-frame global force vector ``(..., C)``."""
        chans = x.shape[-1]
        mean = torch.tensor(self.mean, dtype=x.dtype, device=x.device)
        scale = torch.tensor(self.scale, dtype=x.dtype, device=x.device).clamp_min(_EPS)
        if len(self.scale) != chans:
            raise ValueError(f"ForceNormalizer has {len(self.scale)} channels but got {chans}.")
        return x * scale + mean

    def to_dict(self) -> Dict:
        return {
            "scale": list(self.scale),
            "mean": list(self.mean),
            "quantile": self.quantile,
            "population": self.population,
            "source": self.source,
        }

    @classmethod
    def from_dict(cls, raw: Optional[Dict]) -> Optional["ForceNormalizer"]:
        if not raw:
            return None
        scale = raw.get("scale")
        if not scale:
            return None
        return cls(
            scale=tuple(scale),
            mean=tuple(raw.get("mean") or ()),
            quantile=raw.get("quantile"),
            population=raw.get("population", "contact"),
            source=raw.get("source", ""),
        )

    def describe(self) -> str:
        scale = ", ".join(f"{name}={value:.6g}" for name, value in zip(CHANNELS, self.scale))
        tag = f"q{self.quantile:g}/{self.population}" if self.quantile is not None else "explicit"
        return f"ForceNormalizer[{tag}] scale({scale})"


def normalizer_from_data_config(cfg) -> ForceNormalizer:
    """Build the normalizer a dataset should apply, from its data config.

    ``force_mean``/``force_std`` remain the single source of truth at dataset level;
    the quantile machinery only decides what gets written into them.
    """
    chans = int(getattr(cfg, "force_chans", len(CHANNELS)))
    scale = getattr(cfg, "force_std", None) or [1.0] * chans
    mean = getattr(cfg, "force_mean", None) or [0.0] * chans
    return ForceNormalizer(
        scale=tuple(scale),
        mean=tuple(mean),
        quantile=getattr(cfg, "force_norm_quantile", None) if getattr(cfg, "force_norm_mode", "none") != "none" else None,
        population=getattr(cfg, "force_norm_population", "contact"),
        source=str(getattr(cfg, "force_stats", "") or "config"),
    )


def apply_force_norm_to_data_config(
    cfg,
    stats_path: Optional[str] = None,
    stats: Optional[ForceStats] = None,
) -> ForceNormalizer:
    """Resolve ``force_norm_mode`` into concrete ``force_mean``/``force_std`` on ``cfg``.

    Records the full quantile table on ``cfg.force_norm_stats`` so it lands in the
    checkpoint via ``data_cfg.__dict__`` and can be read back at test time.
    Returns the normalizer that the dataset will apply.
    """
    import json

    mode = str(getattr(cfg, "force_norm_mode", "none") or "none").lower()
    if mode == "none":
        return normalizer_from_data_config(cfg)
    if mode not in {"quantile"}:
        raise ValueError(f"Unknown force_norm_mode `{mode}`. Use `none` or `quantile`.")

    path = stats_path or getattr(cfg, "force_stats", None)
    if stats is None:
        if not path:
            raise ValueError(
                "force_norm_mode=`quantile` requires data.force_stats (or --force-stats) "
                "pointing at a JSON produced by scripts/compute_force_stats.py."
            )
        with open(path, "r", encoding="utf-8") as f:
            stats = ForceStats.from_dict(json.load(f))

    quantile = float(getattr(cfg, "force_norm_quantile", 0.99))
    population = str(getattr(cfg, "force_norm_population", "contact"))
    normalizer = ForceNormalizer.from_stats(stats, quantile, population, source=str(path or "inline"))

    cfg.force_mean = list(normalizer.mean)
    cfg.force_std = list(normalizer.scale)
    cfg.force_norm_stats = stats.to_dict()
    return normalizer
