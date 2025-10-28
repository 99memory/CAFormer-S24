"""Building blocks for the CAFormer architecture."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn


class ConvBNAct(nn.Sequential):
    """A convenience block of convolution, batch normalisation and SiLU."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: Optional[int] = None,
        groups: int = 1,
    ) -> None:
        if padding is None:
            padding = kernel_size // 2
        super().__init__(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride=stride,
                padding=padding,
                groups=groups,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
        )


class SqueezeExcite(nn.Module):
    """Channel attention via squeeze-and-excitation."""

    def __init__(self, channels: int, reduction: int = 4) -> None:
        super().__init__()
        hidden = max(channels // reduction, 4)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, hidden, 1)
        self.act = nn.SiLU(inplace=True)
        self.fc2 = nn.Conv2d(hidden, channels, 1)
        self.gate = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = self.pool(x)
        scale = self.fc1(scale)
        scale = self.act(scale)
        scale = self.fc2(scale)
        scale = self.gate(scale)
        return x * scale


class LocalPerceptionUnit(nn.Module):
    """Depth-wise separable convolution with channel attention."""

    def __init__(self, channels: int, expansion: int = 4) -> None:
        super().__init__()
        hidden = channels * expansion
        self.pw1 = ConvBNAct(channels, hidden, kernel_size=1, padding=0)
        self.dw = ConvBNAct(hidden, hidden, kernel_size=3, groups=hidden)
        self.se = SqueezeExcite(hidden)
        self.pw2 = ConvBNAct(hidden, channels, kernel_size=1, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.pw1(x)
        x = self.dw(x)
        x = self.se(x)
        x = self.pw2(x)
        return x + residual


class PatchEmbed(nn.Module):
    """Convert convolutional features into transformer tokens."""

    def __init__(self, in_channels: int, embed_dim: int, patch_size: int) -> None:
        super().__init__()
        self.proj = nn.Conv2d(in_channels, embed_dim, patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        return self.norm(x)


class MLP(nn.Module):
    """Transformer feed-forward network."""

    def __init__(self, embed_dim: int, mlp_ratio: float, drop: float) -> None:
        super().__init__()
        hidden_dim = int(embed_dim * mlp_ratio)
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, embed_dim)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


@dataclass
class AttentionConfig:
    embed_dim: int
    num_heads: int
    drop: float = 0.0
    attn_drop: float = 0.0


class TransformerBlock(nn.Module):
    """Multi-head self-attention block with residual connections."""

    def __init__(self, cfg: AttentionConfig, mlp_ratio: float = 4.0) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(cfg.embed_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=cfg.embed_dim,
            num_heads=cfg.num_heads,
            dropout=cfg.attn_drop,
            batch_first=True,
        )
        self.drop_path = nn.Dropout(cfg.drop)
        self.norm2 = nn.LayerNorm(cfg.embed_dim)
        self.mlp = MLP(cfg.embed_dim, mlp_ratio=mlp_ratio, drop=cfg.drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.norm1(x)
        attn_out, _ = self.attn(x, x, x, need_weights=False)
        x = residual + self.drop_path(attn_out)
        residual = x
        x = self.norm2(x)
        x = residual + self.drop_path(self.mlp(x))
        return x


class TransformerEncoder(nn.Module):
    """Stack of :class:`TransformerBlock`."""

    def __init__(
        self,
        depth: int,
        cfg: AttentionConfig,
        mlp_ratio: float = 4.0,
    ) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            TransformerBlock(cfg, mlp_ratio=mlp_ratio) for _ in range(depth)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
        return x


class GlobalAggregator(nn.Module):
    """Combine transformer tokens into a single representation."""

    def __init__(self, embed_dim: int, use_cls_token: bool = True) -> None:
        super().__init__()
        self.use_cls_token = use_cls_token
        if use_cls_token:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
            nn.init.trunc_normal_(self.cls_token, std=0.02)
        else:
            self.register_parameter("cls_token", None)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_cls_token:
            cls_token = self.cls_token.expand(x.size(0), -1, -1)
            x = torch.cat((cls_token, x), dim=1)
        x = self.norm(x)
        if self.use_cls_token:
            return x[:, 0]
        return x.mean(dim=1)


class SpatialPyramidPooling(nn.Module):
    """Global summarisation of convolutional features."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.proj = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(x)
        return self.proj(x)


class FusionHead(nn.Module):
    """Fuse convolutional and transformer representations."""

    def __init__(self, conv_dim: int, embed_dim: int, num_classes: int) -> None:
        super().__init__()
        self.norm_conv = nn.LayerNorm(conv_dim)
        self.norm_embed = nn.LayerNorm(embed_dim)
        self.fc = nn.Linear(conv_dim + embed_dim, num_classes)

    def forward(self, conv_feat: torch.Tensor, embed_feat: torch.Tensor) -> torch.Tensor:
        conv_feat = self.norm_conv(conv_feat)
        embed_feat = self.norm_embed(embed_feat)
        fused = torch.cat([conv_feat, embed_feat], dim=-1)
        return self.fc(fused)
