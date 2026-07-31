"""JEPA predictor for NeoForce: masked latent prediction on top of the shared trunk.

Incremental over `encoder.py`:
  - `NeoForceJEPA` subclasses `NeoForceEncoder` and adds a lightweight
    JEPA predictor. The trunk forward is unchanged: masked patches are still replaced
    by `mask_token` and the full token set goes through the trunk.
  - The predictor takes the student's trunk patch tokens; masked positions are replaced
    by a learnable query (+ position and modality embeddings) so the prediction has to
    come from context, then regresses the EMA teacher's patch latents at those positions.
  - `multiblock_masks` implements I-JEPA style contiguous multi-block masking, replacing
    the random scatter mask of the baseline.
"""

import math
from typing import Tuple

import torch
from torch import nn
from torch.nn.init import trunc_normal_

from .dino_layers import TransformerBlock, init_vit_weights, interpolate_patch_pos
from .encoder import NeoForceConfig, NeoForceEncoder


def multiblock_masks(
    batch: int,
    frames: int,
    grid: Tuple[int, int],
    num_blocks: int,
    block_scale: Tuple[float, float],
    block_aspect: Tuple[float, float],
    device: torch.device,
) -> torch.Tensor:
    """I-JEPA style mask: the union of `num_blocks` random rectangles, True = masked."""

    gh, gw = grid
    mask = torch.zeros(batch, frames, gh, gw, dtype=torch.bool, device=device)
    for b in range(batch):
        for k in range(frames):
            for _ in range(num_blocks):
                scale = block_scale[0] + torch.rand(1).item() * (block_scale[1] - block_scale[0])
                aspect = block_aspect[0] + torch.rand(1).item() * (block_aspect[1] - block_aspect[0])
                area = scale * gh * gw
                h = min(gh, max(1, round(math.sqrt(area * aspect))))
                w = min(gw, max(1, round(math.sqrt(area / aspect))))
                top = torch.randint(0, gh - h + 1, (1,)).item()
                left = torch.randint(0, gw - w + 1, (1,)).item()
                mask[b, k, top : top + h, left : left + w] = True
    return mask.reshape(batch, frames, gh * gw)


class JEPAPredictor(nn.Module):
    """Narrow transformer that regresses teacher patch latents at masked positions.

    Masked positions are replaced by a learnable query before the blocks, so the
    prediction must be assembled from the surrounding context tokens. The modality
    embedding makes the predictor shareable across vision/touch (and is the hook for
    the P2 cross-modal extension).
    """

    def __init__(
        self,
        embed_dim: int,
        predictor_dim: int,
        depth: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        num_modalities: int = 2,
    ):
        super().__init__()
        self.input_proj = nn.Linear(embed_dim, predictor_dim)
        self.pos_proj = nn.Linear(embed_dim, predictor_dim)
        self.mask_query = nn.Parameter(torch.zeros(1, 1, predictor_dim))
        self.modality_embed = nn.Embedding(num_modalities, predictor_dim)
        self.blocks = nn.ModuleList(
            [TransformerBlock(predictor_dim, num_heads, mlp_ratio) for _ in range(depth)]
        )
        self.norm = nn.LayerNorm(predictor_dim, eps=1e-6)
        self.output_proj = nn.Linear(predictor_dim, embed_dim)
        self.apply(init_vit_weights)
        trunc_normal_(self.mask_query, std=0.02)

    def forward(
        self,
        tokens: torch.Tensor,
        pos: torch.Tensor,
        patch_mask: torch.Tensor,
        modality: int,
    ) -> torch.Tensor:
        # tokens (B, N, embed_dim), pos (1, N, embed_dim), patch_mask (B, N) True = masked
        x = self.input_proj(tokens)
        query = self.mask_query.to(x.dtype).expand_as(x)
        x = torch.where(patch_mask.unsqueeze(-1), query, x)
        x = x + self.pos_proj(pos).to(x.dtype) + self.modality_embed.weight[modality].view(1, 1, -1)
        for block in self.blocks:
            x = block(x)
        return self.output_proj(self.norm(x))


class NeoForceJEPA(NeoForceEncoder):
    """Baseline encoder + JEPA predictor. Trunk forward is inherited unchanged."""

    def __init__(self, cfg: NeoForceConfig = NeoForceConfig()):
        super().__init__(cfg)
        self.jepa_predictor = JEPAPredictor(
            cfg.embed_dim,
            cfg.predictor_dim,
            cfg.predictor_depth,
            cfg.predictor_heads,
            mlp_ratio=cfg.mlp_ratio,
        )

    def jepa_predict(
        self,
        patch_tokens: torch.Tensor,
        patch_mask: torch.Tensor,
        grid: Tuple[int, int],
        modality: int,
    ) -> torch.Tensor:
        # patch_tokens (B, K, N, D) from forward(); returns (B, K, N, D) predictions.
        b, k, n, d = patch_tokens.shape
        pos = interpolate_patch_pos(self.patch_pos_embed, grid).to(dtype=patch_tokens.dtype)
        pred = self.jepa_predictor(
            patch_tokens.reshape(b * k, n, d),
            pos,
            patch_mask.reshape(b * k, n).to(dtype=torch.bool),
            modality,
        )
        return pred.reshape(b, k, n, -1)
