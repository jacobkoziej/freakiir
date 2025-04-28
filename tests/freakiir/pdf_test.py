# SPDX-License-Identifier: MPL-2.0
#
# pdf_test.py -- probability density function tests
# Copyright (C) 2025  Jacob Koziej <jacobkoziej@gmail.com>

import pytest
import torch

from typing import Optional

from pytest import FixtureRequest
from torch import (
    Generator,
    pi,
)

from freakiir.pdf import (
    Uniform,
)


@pytest.fixture(params=[None, torch.float32, torch.float64])
def dtype(request: FixtureRequest) -> Optional[torch.dtype]:
    return request.param


@pytest.mark.parametrize(
    "r_a, r_b, theta_a, theta_b",
    [
        (0.1, 0.9, 0.0, pi),
        (0.3, 1.6, -pi / 4, 3 * pi / 4),
    ],
)
def test_uniform(
    r_a: float,
    r_b: float,
    theta_a: float,
    theta_b: float,
    dtype: torch.dtype,
    generator: Generator,
    samples: int,
) -> None:
    pdf = Uniform(
        r_a=r_a,
        r_b=r_b,
        theta_a=theta_a,
        theta_b=theta_b,
        dtype=dtype,
        generator=generator,
    )

    samples = pdf(samples)

    if dtype is None:
        dtype = torch.get_default_dtype()

    assert samples.real.dtype == dtype
    assert samples.imag.dtype == dtype

    r = samples.abs()

    assert torch.all(r >= r_a)
    assert torch.all(r <= r_b)

    theta = samples.angle()

    assert torch.all(theta >= theta_a)
    assert torch.all(theta <= theta_b)
