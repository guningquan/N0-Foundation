import math
from typing import Optional, Tuple

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.init import trunc_normal_
from torch.nn.utils import weight_norm


class PatchEmbed(nn.Module):
    """DINOv2-style conv patch embedding."""

    def __init__(self, in_chans: int, embed_dim: int, patch_size: int):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Tuple[int, int]]:
        x = self.proj(x)
        grid = (x.shape[-2], x.shape[-1])
        x = x.flatten(2).transpose(1, 2)
        return x, grid


class MLP(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, out_dim: Optional[int] = None, dropout: float = 0.0):
        super().__init__()
        out_dim = out_dim or dim
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, out_dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        return self.drop(x)


class Attention(nn.Module):
    def __init__(self, dim: int, num_heads: int, qkv_bias: bool = True, attn_dropout: float = 0.0, proj_dropout: float = 0.0):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"embed_dim={dim} must be divisible by num_heads={num_heads}")
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_dropout)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, n, c = x.shape
        qkv = self.qkv(x).reshape(b, n, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(b, n, c)
        return self.proj_drop(self.proj(x))


class LayerScale(nn.Module):
    """DINOv2 LayerScale with checkpoint-compatible parameter names."""

    def __init__(self, dim: int, init_values: float = 1.0):
        super().__init__()
        self.gamma = nn.Parameter(torch.full((dim,), init_values))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.gamma.to(dtype=x.dtype)


class TransformerBlock(nn.Module):
    """ViT block matching DINOv2/timm parameter names for checkpoint loading."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        attn_dropout: float = 0.0,
        layer_scale_init_value: float = 1.0,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = Attention(
            dim,
            num_heads,
            attn_dropout=attn_dropout,
            proj_dropout=dropout,
        )
        self.ls1 = LayerScale(dim, layer_scale_init_value)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.mlp = MLP(dim, int(dim * mlp_ratio), dropout=dropout)
        self.ls2 = LayerScale(dim, layer_scale_init_value)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.ls1(self.attn(self.norm1(x)))
        return x + self.ls2(self.mlp(self.norm2(x)))


class DINOHead(nn.Module):
    """DINOv2 projection head: MLP, L2 norm, weight-normalized prototypes."""

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        use_bn: bool = False,
        nlayers: int = 3,
        hidden_dim: int = 2048,
        bottleneck_dim: int = 256,
        mlp_bias: bool = True,
    ):
        super().__init__()
        nlayers = max(nlayers, 1)
        self.mlp = self._build_mlp(nlayers, in_dim, bottleneck_dim, hidden_dim, use_bn, mlp_bias)
        self.apply(self._init_weights)
        self.last_layer = weight_norm(nn.Linear(bottleneck_dim, out_dim, bias=False))
        self.last_layer.weight_g.data.fill_(1)

    @staticmethod
    def _build_mlp(
        nlayers: int,
        in_dim: int,
        bottleneck_dim: int,
        hidden_dim: int,
        use_bn: bool,
        bias: bool,
    ) -> nn.Module:
        if nlayers == 1:
            return nn.Linear(in_dim, bottleneck_dim, bias=bias)
        layers = [nn.Linear(in_dim, hidden_dim, bias=bias)]
        if use_bn:
            layers.append(nn.BatchNorm1d(hidden_dim))
        layers.append(nn.GELU())
        for _ in range(nlayers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim, bias=bias))
            if use_bn:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.GELU())
        layers.append(nn.Linear(hidden_dim, bottleneck_dim, bias=bias))
        return nn.Sequential(*layers)

    @staticmethod
    def _init_weights(m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.mlp(x)
        eps = 1e-6 if x.dtype in (torch.float16, torch.bfloat16) else 1e-12
        x = F.normalize(x, dim=-1, p=2, eps=eps)
        return self.last_layer(x)


def init_vit_weights(module: nn.Module) -> None:
    if isinstance(module, nn.Linear):
        trunc_normal_(module.weight, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Conv2d):
        fan_out = module.kernel_size[0] * module.kernel_size[1] * module.out_channels
        nn.init.normal_(module.weight, mean=0.0, std=math.sqrt(2.0 / fan_out))
        if module.bias is not None:
            nn.init.zeros_(module.bias)


def interpolate_patch_pos(pos_embed: torch.Tensor, grid: Tuple[int, int]) -> torch.Tensor:
    """Interpolate a square learned patch position table to an arbitrary HxW grid."""

    n = pos_embed.shape[1]
    dim = pos_embed.shape[-1]
    side = int(math.sqrt(n))
    if side * side != n:
        return F.interpolate(pos_embed.transpose(1, 2), size=grid[0] * grid[1], mode="linear").transpose(1, 2)
    pos = pos_embed.reshape(1, side, side, dim).permute(0, 3, 1, 2)
    pos = F.interpolate(pos, size=grid, mode="bicubic", align_corners=False)
    return pos.permute(0, 2, 3, 1).reshape(1, grid[0] * grid[1], dim)
