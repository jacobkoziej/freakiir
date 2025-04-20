# SPDX-License-Identifier: MPL-2.0
#
# pdf.py -- probability density functions
# Copyright (C) 2025  Jacob Koziej <jacobkoziej@gmail.com>

import torch

from collections.abc import Callable
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


def uniform(
    *,
    r_a: float = 0.0,
    r_b: float = 1.0,
    epsilon: float = _EPSILON,
    theta_a: float = 0.0,
    theta_b: float = pi,
    dtype: Optional[dtype] = None,
    generator: Optional[Generator] = None,
) -> Pdf:
    r_a += epsilon
    r_b -= epsilon

    assert r_a <= r_b
    assert theta_a <= theta_b

    def sample(a: float, b: float, samples: int) -> Tensor:
        return (a - b) * torch.rand(
            samples,
            dtype=dtype,
            generator=generator,
        ) + b

    def uniform(samples: int) -> Tensor:
        r = sample(r_a, r_b, samples)
        theta = sample(theta_a, theta_b, samples)

        return r * torch.exp(1j * theta)

    return uniform
