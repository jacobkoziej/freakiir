# SPDX-License-Identifier: MPL-2.0
#
# pdf_test.py -- probability density function tests
# Copyright (C) 2025  Jacob Koziej <jacobkoziej@gmail.com>

import pytest
import torch

from typing import (
    Final,
    Optional,
)

from _pytest.fixtures import SubRequest
from torch import (
    Generator,
    pi,
)

from freakiir.pdf import uniform


SAMPLES: Final[int] = 1 << 12
SEED: Final[int] = 0xF14CE2E4


@pytest.fixture(params=[None, torch.float32, torch.float64])
def dtype(request: SubRequest) -> Optional[torch.dtype]:
    return request.param


@pytest.fixture
def generator() -> Generator:
    return Generator().manual_seed(SEED)


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
) -> None:
    pdf = uniform(
        r_a=r_a,
        r_b=r_b,
        theta_a=theta_a,
        theta_b=theta_b,
        dtype=dtype,
        generator=generator,
    )

    samples = pdf(SAMPLES)

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
