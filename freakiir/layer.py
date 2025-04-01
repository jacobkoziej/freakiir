# SPDX-License-Identifier: MPL-2.0
#
# layer.py -- model layers
# Copyright (C) 2025  Jacob Koziej <jacobkoziej@gmail.com>

import torch
import torch.nn as nn

from dataclasses import dataclass

from torch import Tensor


class ComplexToReal(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        return torch.view_as_real(x)


@dataclass(frozen=True)
class MlpConfig:
    in_features: int
    hidden_features: int
    out_features: int
    hidden_layers: int

    def __post_init__(self) -> None:
        assert self.in_features > 0
        assert self.hidden_features > 0
        assert self.out_features > 0
        assert self.hidden_layers > 0


class Mlp(nn.Module):
    def __init__(self, config: MlpConfig):
        super().__init__()

        def gen_layer(in_features, out_features):
            return nn.Sequential(
                nn.Linear(in_features, out_features),
                nn.LayerNorm(out_features),
                nn.LeakyReLU(),
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
        return torch.view_as_complex(x)


class ReflectIntoComplexUnitCircle(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        r = x.abs()
        theta = x.angle()

        r = torch.where(r <= 1, r, 1 / r)

        return r * torch.exp(1j * theta)
