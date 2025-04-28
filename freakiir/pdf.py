# SPDX-License-Identifier: MPL-2.0
#
# pdf.py -- probability density functions
# Copyright (C) 2025  Jacob Koziej <jacobkoziej@gmail.com>

import torch

from abc import (
    ABC,
    abstractmethod,
)
from collections.abc import Callable
from dataclasses import dataclass
from typing import (
    Final,
    NewType,
    Optional,
)

from torch import (
    Generator,
    Tensor,
    dtype,
    pi,
)

_EPSILON: Final[float] = 1e-6

Pdf = NewType("Pdf", Callable[[int], Tensor])


class PDF(ABC):
    @abstractmethod
    def __call__(self, samples: int) -> Tensor: ...


@dataclass(kw_only=True)
class Uniform(PDF):
    r_a: float = 0.0
    r_b: float = 1.0
    epsilon: float = _EPSILON
    theta_a: float = 0.0
    theta_b: float = pi
    dtype: Optional[dtype] = None
    generator: Optional[Generator] = None

    def __call__(self, samples: int) -> Tensor:
        r_a = self.r_a + self.epsilon
        r_b = self.r_b - self.epsilon

        def sample(a: float, b: float, samples: int) -> Tensor:
            return (a - b) * torch.rand(
                samples,
                dtype=self.dtype,
                generator=self.generator,
            ) + b

        r = sample(r_a, r_b, samples)
        theta = sample(self.theta_a, self.theta_b, samples)

        return r * torch.exp(1j * theta)
