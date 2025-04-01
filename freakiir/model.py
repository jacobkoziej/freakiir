# SPDX-License-Identifier: MPL-2.0
#
# model.py -- model components
# Copyright (C) 2025  Jacob Koziej <jacobkoziej@gmail.com>

import torch
import torch.nn as nn
import torch.optim as optim

from abc import ABC
from dataclasses import dataclass
from typing import TypeAlias

from einops import reduce
from pytorch_lightning import LightningModule
from torch import Tensor

from .dsp import polyvalfromroots
from .layer import (
    ConstructMinimumPhaseSections,
    DecibelMagnitude,
    Mlp,
    MlpConfig,
    RealToComplex,
    ReflectIntoComplexUnitCircle,
)


ModelStepInput: TypeAlias = tuple[Tensor, Tensor]


@dataclass(frozen=True, kw_only=True)
class ModelConfig:
    inputs: int = 512
    sections: int = 4
    hidden_features: int = 2048
    hidden_layers: int = 4

    down_order: bool = False


@dataclass(frozen=True, kw_only=True)
class ModelInput:
    w: Tensor
    h: Tensor


@dataclass(frozen=True, kw_only=True)
class ModelOutput:
    z: Tensor
    p: Tensor
    k: Tensor


class Base(LightningModule, ABC):
    layers: nn.modules.container.Sequential

    def __init__(self) -> None:
        super().__init__()

        self.loss = nn.MSELoss()

    def _output2prediction(
        w: Tensor,
        z: Tensor,
        p: Tensor,
        k: Tensor,
    ) -> Tensor:
        h = torch.exp(1j * w)

        h = k * polyvalfromroots(h, z) / polyvalfromroots(h, p)
        h = reduce(h, "... sections h -> ... h", "prod")

        return h

    def _step(
        self,
        batch: ModelStepInput,
        batch_idx: int,
        phase: str,
    ) -> Tensor:
        w, h = batch

        z, p, k = self.forward(h)

        h_predicted = self._output2prediction(w, z, p, k)

        loss = self.loss(
            self.preprocess(h_predicted),
            self.preprocess(h),
        )

        self.log(f"{phase}/loss", loss, prog_bar=True)

        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters())

        return optimizer

    def forward(self, h: Tensor) -> Tensor:
        z, p, k = self.layers(h)

        return z, p, k

    def predict(self, input: ModelInput) -> ModelOutput:
        z, p, k = self.forward(input.h)

        return ModelOutput(z=z, p=p, k=k)

    def test_step(self, batch: ModelStepInput, batch_idx: int) -> Tensor:
        return self._step(batch, batch_idx, "test")

    def training_step(self, batch: ModelStepInput, batch_idx: int) -> Tensor:
        return self._step(batch, batch_idx, "train")

    def validation_step(self, batch: ModelStepInput, batch_idx: int) -> Tensor:
        return self._step(batch, batch_idx, "val")


class MinimumPhase(Base):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()

        self.save_hyperparameters()

        mlp_config = MlpConfig(
            in_features=config.inputs,
            hidden_features=config.hidden_features,
            out_features=4 * config.sections,
            hidden_layers=config.hidden_layers,
        )

        self.layers = nn.Sequential(
            DecibelMagnitude(),
            Mlp(mlp_config),
            RealToComplex(),
            ReflectIntoComplexUnitCircle(),
            ConstructMinimumPhaseSections(config.sections, config.down_order),
        )
