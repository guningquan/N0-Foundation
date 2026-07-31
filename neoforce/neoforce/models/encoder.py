"""NeoForce encoder: a joint visual-tactile ViT over RGB frames and force fields.

RGB frames and dual-sensor force fields are patched onto a shared token grid and run
through one trunk, so the two modalities attend to each other directly rather than
being fused after the fact. A temporal transformer over per-frame CLS pairs supplies
context, and a progressive decoder reconstructs the force field and contact mask at the input
resolution, supervised at every scale it passes through.

Two properties of the tactile signal shape the design:

* **Force values are tiny.** The tactile conversion model emits fz = pressure x 1e-3 with a
  pressure threshold of 1e-2, so in-contact forces start around 1e-5 N. Fed to a Huber
  loss with the usual beta=1 they sit deep in the quadratic region, where the gradient
  is proportional to the error itself and the dense branch is starved next to the
  distillation terms. The dataset therefore rescales the target per channel (see
  ``data/force_norm.py``); the model always works in those normalized units.
* **Zero means "no contact".** Non-contact pixels are exact zeros, the touch-dropout
  path zeroes whole frames, and the vision-only path feeds an all-zero tactile tensor.
  Normalization is scale-only for exactly this reason, and the tactile stem encodes
  "missing" as a source embedding rather than as a magnitude.
"""

import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.init import trunc_normal_

from .dino_layers import DINOHead, PatchEmbed, TransformerBlock, init_vit_weights, interpolate_patch_pos

BACKBONE_SPECS = {
    "vit-s": {"embed_dim": 384, "depth": 12, "num_heads": 6},
    "vit-b": {"embed_dim": 768, "depth": 12, "num_heads": 12},
    "vit-l": {"embed_dim": 1024, "depth": 24, "num_heads": 16},
    "vit-g": {"embed_dim": 1536, "depth": 40, "num_heads": 24},
}


def _canonical_backbone_key(backbone: str) -> str:
    token = backbone.split("-img", 1)[0].split("-patch", 1)[0].lower().replace("_", "-")
    return token.replace("vit", "vit-", 1) if token in {"vits", "vitb", "vitl", "vitg"} else token


@dataclass
class NeoForceConfig:
    """Model configuration.

    ``backbone`` is a shorthand that fills in the rest: ``ViT-B-img360x640-patch20``
    sets the width/depth/heads from the ViT size and the resolution and patch size from
    the suffixes, so a run is described by one string.
    """

    backbone: str = "ViT-B-img360x640-patch20"
    image_size: Tuple[int, int] = (360, 640)
    patch_size: int = 20
    in_chans: int = 3
    force_chans: int = 6
    mask_chans: int = 6
    embed_dim: int = 768
    depth: int = 12
    num_heads: int = 12
    mlp_ratio: float = 4.0
    temporal_depth: int = 4
    temporal_heads: int = 12
    temporal_max_frames: int = 8
    proj_dim: int = 256
    prototype_dim: int = 8192
    dropout: float = 0.0

    # Decoder. Decoding starts at out_size/8 and doubles to the full resolution, one thin
    # conv block per stage, with a prediction head and a loss at every stage. A token-grid
    # map stretched 20x in one bilinear jump cannot represent a contact boundary narrower
    # than a patch; going up in steps, each with learnable weights at the new resolution,
    # can. Deep supervision keeps the coarse scale anchoring the fine ones instead of
    # leaving the finest stage to discover the structure on its own.
    decoder_start_stride: int = 8
    decoder_stage_channels: Tuple[int, ...] = (128, 64, 32, 16)
    decoder_deep_supervision: bool = True

    # Tactile maps are kept at the RGB resolution; the tactile stem uses the visual
    # patch size so both modalities land on the same token grid.
    tactile_full_resolution: bool = True
    downsample: int = 1

    # JEPA predictor (training only; unused at inference).
    predictor_dim: int = 384
    predictor_depth: int = 6
    predictor_heads: int = 12
    jepa_num_blocks: int = 4
    jepa_block_scale: Tuple[float, float] = (0.15, 0.2)
    jepa_block_aspect: Tuple[float, float] = (0.75, 1.5)

    def __post_init__(self):
        for name in ("image_size", "decoder_stage_channels", "jepa_block_scale", "jepa_block_aspect"):
            value = getattr(self, name)
            if isinstance(value, list):
                setattr(self, name, tuple(value))
        if self.tactile_full_resolution:
            self.downsample = 1
        if not self.backbone:
            return
        key = _canonical_backbone_key(self.backbone)
        if key not in BACKBONE_SPECS:
            raise ValueError(f"Unknown backbone `{self.backbone}`. Supported: ViT-S, ViT-B, ViT-L, ViT-G.")
        spec = BACKBONE_SPECS[key]
        self.embed_dim = spec["embed_dim"]
        self.depth = spec["depth"]
        self.num_heads = spec["num_heads"]
        self.temporal_heads = spec["num_heads"]

        image_match = re.search(r"img(\d+)x(\d+)", self.backbone.lower())
        if image_match:
            self.image_size = (int(image_match.group(1)), int(image_match.group(2)))
        patch_match = re.search(r"patch(\d+)", self.backbone.lower())
        if patch_match:
            self.patch_size = int(patch_match.group(1))
            self.downsample = 1 if self.tactile_full_resolution else self.patch_size


def stage_strides(start_stride: int) -> Tuple[int, ...]:
    """``8 -> (8, 4, 2, 1)``: the ladder from ``out_size/start`` up to ``out_size``."""
    if start_stride < 1 or (start_stride & (start_stride - 1)) != 0:
        raise ValueError(f"decoder_start_stride must be a positive power of two, got {start_stride}")
    strides, stride = [], int(start_stride)
    while stride >= 1:
        strides.append(stride)
        stride //= 2
    return tuple(strides)


def stage_size(out_size: Tuple[int, int], stride: int) -> Tuple[int, int]:
    return (max(1, -(-out_size[0] // stride)), max(1, -(-out_size[1] // stride)))


class PredictionHeads(nn.Module):
    """1x1 heads producing the force field and the contact-mask logits at one scale."""

    def __init__(self, channels: int, force_chans: int, mask_chans: int):
        super().__init__()
        self.force = nn.Conv2d(channels, force_chans, kernel_size=1)
        self.mask = nn.Conv2d(channels, mask_chans, kernel_size=1)

    def forward(self, feat: torch.Tensor) -> Dict[str, torch.Tensor]:
        return {"force": self.force(feat), "mask_logits": self.mask(feat)}


class ProgressiveDecoder(nn.Module):
    """Decode from ``out_size/8`` up to ``out_size``, predicting at every stage.

    Channel width shrinks as resolution grows: a 16-channel block at 360x640 costs about
    what a 128-channel block costs at 45x80, so the ladder stays affordable at full
    resolution. Each stage carries its own prediction head, and with deep supervision the
    intermediate ones are trained directly rather than only through the final output.
    """

    def __init__(
        self,
        embed_dim: int,
        force_chans: int,
        mask_chans: int,
        start_stride: int = 8,
        stage_channels: Sequence[int] = (128, 64, 32, 16),
        deep_supervision: bool = True,
    ):
        super().__init__()
        self.strides = stage_strides(start_stride)
        if len(stage_channels) != len(self.strides):
            raise ValueError(
                f"decoder_stage_channels has {len(stage_channels)} entries but "
                f"decoder_start_stride={start_stride} needs {len(self.strides)} (strides {self.strides})."
            )
        self.stage_channels = tuple(int(c) for c in stage_channels)
        self.deep_supervision = bool(deep_supervision)
        self.entry = nn.Conv2d(embed_dim, self.stage_channels[0], kernel_size=1)

        blocks = []
        for index, channels in enumerate(self.stage_channels):
            in_channels = self.stage_channels[0] if index == 0 else self.stage_channels[index - 1]
            layers = [nn.Conv2d(in_channels, channels, kernel_size=3, padding=1), nn.GroupNorm(1, channels), nn.GELU()]
            if index == 0:
                # The coarsest stage carries the most semantic work and is the cheapest
                # place to spend a second conv; finer stages stay deliberately thin.
                layers += [nn.Conv2d(channels, channels, kernel_size=3, padding=1), nn.GroupNorm(1, channels), nn.GELU()]
            blocks.append(nn.Sequential(*layers))
        self.blocks = nn.ModuleList(blocks)

        # Heads only where they are supervised: an unsupervised head would be dead weight
        # and would trip DDP's unused-parameter detection.
        last = len(self.stage_channels) - 1
        self.heads = nn.ModuleList(
            PredictionHeads(channels, force_chans, mask_chans)
            if (self.deep_supervision or index == last) else nn.Identity()
            for index, channels in enumerate(self.stage_channels)
        )

    def forward(
        self,
        patch_tokens: torch.Tensor,
        grid: Tuple[int, int],
        out_size: Tuple[int, int],
    ) -> Dict[str, torch.Tensor]:
        b, _, d = patch_tokens.shape
        feat = self.entry(patch_tokens.transpose(1, 2).reshape(b, d, grid[0], grid[1]))

        sizes = [stage_size(out_size, stride) for stride in self.strides]
        sizes[-1] = tuple(out_size)
        aux: List[Dict[str, torch.Tensor]] = []
        final: Dict[str, torch.Tensor] = {}
        for index, size in enumerate(sizes):
            feat = F.interpolate(feat, size=size, mode="bilinear", align_corners=False)
            feat = self.blocks[index](feat)
            head = self.heads[index]
            if isinstance(head, nn.Identity):
                continue
            preds = head(feat)
            if index == len(sizes) - 1:
                final = preds
            else:
                aux.append({"stride": self.strides[index], **preds})
        final["aux"] = aux
        return final


class NeoForceEncoder(nn.Module):
    """Joint visual-tactile encoder.

    Inputs are sequences::

        rgb          (B, K, 3, H, W)   ImageNet-normalized wrist view
        force        (B, K, 6, H, W)   [left fx,fy,fz, right fx,fy,fz], optional
        mask         (B, K, 6, H, W)   per-axis binary contact mask, optional
        force_global (B, K, 6)         per-frame channel mean, derived from force if omitted

    ``force`` is optional throughout: dropped frames are zero-filled and flagged through
    the source embedding, which is what makes vision-only inference possible when the
    model is trained with touch dropout.
    """

    def __init__(self, cfg: NeoForceConfig = NeoForceConfig()):
        super().__init__()
        self.cfg = cfg
        if cfg.downsample < 1 or cfg.patch_size % cfg.downsample != 0:
            raise ValueError(f"downsample={cfg.downsample} must be a positive divisor of patch_size={cfg.patch_size}")

        self.vision_patch_embed = PatchEmbed(cfg.in_chans, cfg.embed_dim, cfg.patch_size)
        self.touch_patch_embed = PatchEmbed(cfg.force_chans + cfg.mask_chans, cfg.embed_dim, cfg.patch_size // cfg.downsample)

        grid = (cfg.image_size[0] // cfg.patch_size, cfg.image_size[1] // cfg.patch_size)
        self.patch_pos_embed = nn.Parameter(torch.zeros(1, grid[0] * grid[1], cfg.embed_dim))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, cfg.embed_dim))
        self.mask_token = nn.Parameter(torch.zeros(1, 1, cfg.embed_dim))
        self.modality_embed = nn.Embedding(2, cfg.embed_dim)  # 0 visual, 1 tactile
        self.source_embed = nn.Embedding(2, cfg.embed_dim)  # 0 real, 1 missing

        self.blocks = nn.ModuleList(
            TransformerBlock(cfg.embed_dim, cfg.num_heads, cfg.mlp_ratio, dropout=cfg.dropout)
            for _ in range(cfg.depth)
        )
        self.norm = nn.LayerNorm(cfg.embed_dim, eps=1e-6)

        temporal_layer = nn.TransformerEncoderLayer(
            d_model=cfg.embed_dim,
            nhead=cfg.temporal_heads,
            dim_feedforward=int(cfg.embed_dim * cfg.mlp_ratio),
            dropout=cfg.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.temporal_pos_embed = nn.Parameter(torch.zeros(1, cfg.temporal_max_frames * 2, cfg.embed_dim))
        self.temporal = nn.TransformerEncoder(temporal_layer, num_layers=cfg.temporal_depth)
        self.temporal_norm = nn.LayerNorm(cfg.embed_dim, eps=1e-6)

        self.global_embed = nn.Linear(cfg.force_chans, cfg.embed_dim)
        self.global_head = nn.Linear(cfg.embed_dim, cfg.force_chans)
        self.proj_v = nn.Linear(cfg.embed_dim, cfg.proj_dim)
        self.proj_t = nn.Linear(cfg.embed_dim, cfg.proj_dim)
        self.dino_head_v = DINOHead(cfg.embed_dim, cfg.prototype_dim)
        self.dino_head_t = DINOHead(cfg.embed_dim, cfg.prototype_dim)
        self.dino_head_cross = DINOHead(cfg.embed_dim, cfg.prototype_dim)
        self.decoder = ProgressiveDecoder(
            cfg.embed_dim,
            cfg.force_chans,
            cfg.mask_chans,
            start_stride=cfg.decoder_start_stride,
            stage_channels=cfg.decoder_stage_channels,
            deep_supervision=cfg.decoder_deep_supervision,
        )

        self.apply(init_vit_weights)
        trunc_normal_(self.patch_pos_embed, std=0.02)
        trunc_normal_(self.temporal_pos_embed, std=0.02)
        nn.init.normal_(self.cls_token, std=1e-6)
        nn.init.normal_(self.mask_token, std=1e-6)

    def _normalize_sequence(self, x: torch.Tensor, name: str) -> torch.Tensor:
        if x.ndim == 4:
            x = x.unsqueeze(1)
        if x.ndim != 5:
            raise ValueError(f"{name} must have shape (B,K,C,H,W) or (B,C,H,W), got {tuple(x.shape)}")
        return x

    def _make_touch_input(self, force, mask, batch, frames, size, device):
        if force is None:
            force = torch.zeros(batch, frames, self.cfg.force_chans, *size, device=device)
            touch_available = torch.zeros(batch, frames, dtype=torch.bool, device=device)
        else:
            force = self._normalize_sequence(force, "force").to(device=device)
            touch_available = torch.ones(force.shape[0], force.shape[1], dtype=torch.bool, device=device)

        if mask is None:
            mask = torch.zeros(force.shape[0], force.shape[1], self.cfg.mask_chans, force.shape[-2], force.shape[-1], device=device)
        else:
            mask = self._normalize_sequence(mask, "mask").to(device=device)

        if force.shape[-2:] != size:
            flat = force.flatten(0, 1)
            force = F.interpolate(flat, size=size, mode="bilinear", align_corners=False).reshape(force.shape[0], force.shape[1], force.shape[2], *size)
        if mask.shape[-2:] != size:
            flat = mask.flatten(0, 1)
            mask = F.interpolate(flat.float(), size=size, mode="nearest").reshape(mask.shape[0], mask.shape[1], mask.shape[2], *size)
        return torch.cat([force, mask.float()], dim=2), mask.float(), touch_available

    def load_dinov2_weights(self, checkpoint_path: str, strict: bool = False) -> Dict[str, object]:
        """Warm-start the visual stem and trunk from a DINOv2 checkpoint.

        Only shape-compatible tensors are taken, so a DINOv2 trained at a different patch
        size still contributes its trunk while its patch stem is skipped.
        """
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        state = checkpoint.get("model", checkpoint.get("teacher", checkpoint))
        mapped, target_state = {}, self.state_dict()
        target_grid = (self.cfg.image_size[0] // self.cfg.patch_size, self.cfg.image_size[1] // self.cfg.patch_size)
        for key, value in state.items():
            key = key.removeprefix("backbone.")
            if key.startswith("patch_embed."):
                mapped["vision_" + key] = value
            elif key == "pos_embed" and value.ndim == 3 and value.shape[1] > 1:
                mapped["patch_pos_embed"] = interpolate_patch_pos(value[:, 1:].detach(), target_grid)
            elif key == "cls_token" or key.startswith(("blocks.", "norm.")):
                mapped[key] = value
        mapped = {k: v for k, v in mapped.items() if k in target_state and target_state[k].shape == v.shape}
        return self.load_state_dict(mapped, strict=strict)

    def forward(
        self,
        rgb: torch.Tensor,
        force: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        force_global: Optional[torch.Tensor] = None,
        touch_available: Optional[torch.Tensor] = None,
        vision_patch_mask: Optional[torch.Tensor] = None,
        touch_patch_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        rgb = self._normalize_sequence(rgb, "rgb")
        b, k, _, h, w = rgb.shape
        device = rgb.device

        if force_global is None:
            if force is not None:
                force_global = self._normalize_sequence(force, "force").to(device=device).mean(dim=(-2, -1))
            else:
                force_global = torch.zeros(b, k, self.cfg.force_chans, device=device)
        elif force_global.ndim == 2:
            force_global = force_global.unsqueeze(1).to(device=device)
        else:
            force_global = force_global.to(device=device)

        touch_size = (h // self.cfg.downsample, w // self.cfg.downsample)
        touch, _, inferred_available = self._make_touch_input(force, mask, b, k, touch_size, device)
        touch_available = inferred_available if touch_available is None else touch_available.to(device=device, dtype=torch.bool)

        rgb_flat, touch_flat = rgb.flatten(0, 1), touch.flatten(0, 1)
        v_patch, grid = self.vision_patch_embed(rgb_flat)
        t_patch, t_grid = self.touch_patch_embed(touch_flat)
        if t_grid != grid:
            raise RuntimeError(f"vision and tactile grids differ: {grid} vs {t_grid}")

        n = v_patch.shape[1]
        pos = interpolate_patch_pos(self.patch_pos_embed, grid).to(dtype=v_patch.dtype)
        v_patch = v_patch + pos + self.modality_embed.weight[0].view(1, 1, -1) + self.source_embed.weight[0].view(1, 1, -1)
        t_source = self.source_embed((~touch_available).long().flatten()).unsqueeze(1)
        t_patch = t_patch + pos + self.modality_embed.weight[1].view(1, 1, -1) + t_source

        if vision_patch_mask is not None:
            vm = vision_patch_mask.reshape(b * k, n).to(device=device, dtype=torch.bool)
            v_patch = torch.where(vm.unsqueeze(-1), self.mask_token.to(v_patch.dtype), v_patch)
        if touch_patch_mask is not None:
            tm = touch_patch_mask.reshape(b * k, n).to(device=device, dtype=torch.bool)
            t_patch = torch.where(tm.unsqueeze(-1), self.mask_token.to(t_patch.dtype) + t_source, t_patch)

        g_embed = self.global_embed(force_global.flatten(0, 1)).unsqueeze(1)
        cls_v = self.cls_token.expand(b * k, -1, -1) + self.modality_embed.weight[0].view(1, 1, -1)
        cls_t = self.cls_token.expand(b * k, -1, -1) + self.modality_embed.weight[1].view(1, 1, -1) + t_source + g_embed

        x = torch.cat([cls_v, v_patch, cls_t, t_patch], dim=1)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        cls_v, patch_v, cls_t, patch_t = x[:, 0], x[:, 1 : 1 + n], x[:, 1 + n], x[:, 2 + n :]

        temporal_tokens = torch.stack([cls_v.reshape(b, k, -1), cls_t.reshape(b, k, -1)], dim=2).reshape(b, k * 2, -1)
        temporal_tokens = temporal_tokens + self.temporal_pos_embed[:, : k * 2].to(dtype=temporal_tokens.dtype)
        temporal_tokens = self.temporal_norm(self.temporal(temporal_tokens))
        frame_context = temporal_tokens.reshape(b, k, 2, -1).mean(dim=2).reshape(b * k, 1, -1)

        decoded = self.decoder(patch_t + patch_v + frame_context, grid, touch_size)
        out = {
            "force": decoded["force"].reshape(b, k, self.cfg.force_chans, *touch_size),
            "mask_logits": decoded["mask_logits"].reshape(b, k, self.cfg.mask_chans, *touch_size),
            "force_global": self.global_head(frame_context.squeeze(1)).reshape(b, k, self.cfg.force_chans),
            "z_v": F.normalize(self.proj_v(cls_v).reshape(b, k, -1), dim=-1),
            "z_t": F.normalize(self.proj_t(cls_t).reshape(b, k, -1), dim=-1),
            "cls_v": cls_v.reshape(b, k, -1),
            "cls_t": cls_t.reshape(b, k, -1),
            "patch_v": patch_v.reshape(b, k, n, -1),
            "patch_t": patch_t.reshape(b, k, n, -1),
            "dino_v": self.dino_head_v(cls_v).reshape(b, k, -1),
            "dino_t": self.dino_head_t(cls_t).reshape(b, k, -1),
            "dino_cross_v": self.dino_head_cross(cls_v).reshape(b, k, -1),
            "dino_cross_t": self.dino_head_cross(cls_t).reshape(b, k, -1),
            "touch_available": touch_available,
        }
        # Deep-supervision predictions, coarse to fine. NeoForceLoss adds a term per entry
        # when the `aux` weight is set; evaluation ignores them.
        out["aux"] = [
            {
                "stride": e["stride"],
                "force": e["force"].reshape(b, k, self.cfg.force_chans, *e["force"].shape[-2:]),
                "mask_logits": e["mask_logits"].reshape(b, k, self.cfg.mask_chans, *e["mask_logits"].shape[-2:]),
            }
            for e in decoded.get("aux", [])
        ]
        return out
