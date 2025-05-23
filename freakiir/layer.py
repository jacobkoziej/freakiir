# SPDX-License-Identifier: MPL-2.0
#
# layer.py -- model layers
# Copyright (C) 2025  Jacob Koziej <jacobkoziej@gmail.com>

import torch
import torch.nn as nn

from dataclasses import dataclass

from einops import rearrange
from torch import Tensor

from .dsp import (
    construct_sections,
    order_sections,
    unwrap,
)


class ComplexToReal(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        return torch.view_as_real(x)


class ConstructAllPassSections(nn.Module):
    def __init__(self, sections: int, down_order: bool):
        super().__init__()

        self.sections = sections
        self.down_order = down_order

    def forward(self, x: Tensor) -> Tensor:
        sections = self.sections
        down_order = self.down_order

        p = order_sections(x, down_order=down_order)

        r = p.abs()
        theta = p.angle()

        z = (1 / r) * torch.exp(1j * theta)

        z = construct_sections(z, sections, conjugate_pairs=True)
        p = construct_sections(p, sections, conjugate_pairs=True)
        k = (torch.norm(p, dim=-1) / torch.norm(z, dim=-1)).squeeze()

        return z, p, k


class ConstructMinimumPhaseSections(nn.Module):
    def __init__(self, sections: int, down_order: bool):
        super().__init__()

        self.sections = sections
        self.down_order = down_order

    def forward(self, x: Tensor) -> Tensor:
        sections = self.sections
        down_order = self.down_order

        z = order_sections(x[..., :sections], down_order=down_order)
        p = order_sections(x[..., sections:], down_order=down_order)

        z = construct_sections(z, sections, conjugate_pairs=True)
        p = construct_sections(p, sections, conjugate_pairs=True)
        k = torch.ones(z.shape[:-1], dtype=z.real.dtype).to(z.device)

        return z, p, k


class DecibelMagnitude(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        return 20 * torch.log10(x.abs())


@dataclass
class MlpConfig:
    in_features: int
    hidden_features: int
    out_features: int
    hidden_layers: int
    negative_slope: float = 0.2

    def __post_init__(self) -> None:
        assert self.in_features > 0
        assert self.hidden_features > 0
        assert self.out_features > 0
        assert self.hidden_layers > 0
        assert self.negative_slope > 0


class Mlp(nn.Module):
    def __init__(self, config: MlpConfig):
        super().__init__()

        def gen_layer(in_features, out_features):
            return nn.Sequential(
                nn.Linear(in_features, out_features),
                nn.LayerNorm(out_features),
                nn.LeakyReLU(config.negative_slope),
            )

        self.config = config

        self.in_layer = gen_layer(config.in_features, config.hidden_features)

        self.hidden_layers = nn.Sequential(
            *[
                gen_layer(config.hidden_features, config.hidden_features)
                for _ in range(config.hidden_layers)
            ],
        )

        self.out_layer = gen_layer(config.hidden_features, config.out_features)

    def forward(self, x: Tensor) -> Tensor:
        x = self.in_layer.forward(x)
        x = self.hidden_layers.forward(x)
        x = self.out_layer.forward(x)

        return x


class RealToComplex(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        x = rearrange(x, "... (n r) -> ... n r", r=2)

        return torch.view_as_complex(x)


class ReflectIntoComplexUnitCircle(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        r = x.abs()
        theta = x.angle()

        r = torch.where(r <= 1, r, 1 / r)

        return r * torch.exp(1j * theta)


class UnwrapPhase(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        return unwrap(x.angle())
