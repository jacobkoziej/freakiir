# SPDX-License-Identifier: MPL-2.0
#
# model.py -- model components
# Copyright (C) 2025  Jacob Koziej <jacobkoziej@gmail.com>

from dataclasses import dataclass
from typing import TypeAlias

from torch import Tensor


ModelStepInput: TypeAlias = tuple[Tensor, Tensor]


@dataclass(frozen=True, kw_only=True)
class ModelInput:
    w: Tensor
    h: Tensor


@dataclass(frozen=True, kw_only=True)
class ModelOutput:
    z: Tensor
    p: Tensor
    k: Tensor
