"""Implementation of the CAFormer architecture described in the paper."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Tuple

import torch
from torch import nn

from .modules import (
    AttentionConfig,
    ConvBNAct,
    FusionHead,
    GlobalAggregator,
    LocalPerceptionUnit,
    PatchEmbed,
    SpatialPyramidPooling,
    TransformerEncoder,
)


@dataclass
class CAFormerConfig:
    """Configuration for building a CAFormer model."""

    in_channels: int = 3
    stem_channels: Tuple[int, int, int] = (32, 64, 128)
    attention_dims: Tuple[int, ...] = (192, 256, 320)
    attention_heads: Tuple[int, ...] = (3, 4, 5)
    attention_depths: Tuple[int, ...] = (2, 4, 4)
    patch_sizes: Tuple[int, ...] = (2, 2, 2)
    num_classes: int = 7
    dropout: float = 0.1
    mlp_ratio: float = 4.0

    def __post_init__(self) -> None:
        if not (
            len(self.attention_dims)
            == len(self.attention_heads)
            == len(self.attention_depths)
            == len(self.patch_sizes)
        ):
            raise ValueError("Attention configuration tuples must have the same length")


class ConvBackbone(nn.Module):
    """Hierarchical convolutional feature extractor."""

    def __init__(self, in_channels: int, stem_channels: Iterable[int]) -> None:
        super().__init__()
        channels = [in_channels, *stem_channels]
        layers = []
        for idx in range(len(stem_channels)):
            stride = 2 if idx > 0 else 1
            layers.append(
                ConvBNAct(
                    channels[idx],
                    channels[idx + 1],
                    kernel_size=3,
                    stride=stride,
                )
            )
            layers.append(LocalPerceptionUnit(channels[idx + 1]))
        self.body = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.body(x)


class CAFormer(nn.Module):
    """CNN-ViT hybrid for skin lesion classification."""

    def __init__(self, config: CAFormerConfig) -> None:
        super().__init__()
        self.config = config
        self.backbone = ConvBackbone(config.in_channels, config.stem_channels)
        last_cnn_channels = config.stem_channels[-1]

        embed_layers = []
        transformers = []
        for dim, heads, depth, patch in zip(
            config.attention_dims,
            config.attention_heads,
            config.attention_depths,
            config.patch_sizes,
        ):
            embed_layers.append(PatchEmbed(last_cnn_channels, dim, patch))
            transformers.append(
                TransformerEncoder(
                    depth=depth,
                    cfg=AttentionConfig(
                        embed_dim=dim,
                        num_heads=heads,
                        drop=config.dropout,
                        attn_drop=config.dropout / 2,
                    ),
                    mlp_ratio=config.mlp_ratio,
                )
            )
            last_cnn_channels = dim

        self.patch_embeds = nn.ModuleList(embed_layers)
        self.transformers = nn.ModuleList(transformers)
        self.patch_sizes = config.patch_sizes

        fusion_dim = config.attention_dims[-1]
        self.conv_pool = SpatialPyramidPooling(config.stem_channels[-1], fusion_dim)
        self.aggregator = GlobalAggregator(fusion_dim, use_cls_token=True)
        self.head = FusionHead(fusion_dim, fusion_dim, config.num_classes)
        self.dropout = nn.Dropout(config.dropout)

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        conv_features = self.backbone(x)
        tokens = None
        height, width = conv_features.shape[-2:]
        current = conv_features
        for patch_embed, transformer, patch in zip(
            self.patch_embeds, self.transformers, self.patch_sizes
        ):
            tokens = patch_embed(current)
            tokens = transformer(tokens)
            height //= patch
            width //= patch
            current = tokens.transpose(1, 2).reshape(
                x.size(0), tokens.size(-1), height, width
            )

        assert tokens is not None  # for type checkers
        aggregated = self.aggregator(tokens)
        conv_vector = self.conv_pool(conv_features).flatten(1)
        logits = self.head(self.dropout(conv_vector), self.dropout(aggregated))
        return logits


def build_caformer(**kwargs) -> CAFormer:
    """Factory function to instantiate a :class:`CAFormer`."""

    config = CAFormerConfig(**kwargs)
    return CAFormer(config)
