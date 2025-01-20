# SPDX-License-Identifier: MPL-2.0
#
# dsp_test.py -- digital signal processing tests
# Copyright (C) 2025  Jacob Koziej <jacobkoziej@gmail.com>

import numpy as np
import pytest
import torch

from typing import Final

from scipy import signal
from torch import Tensor

from freakiir.dsp import (
    flatten_sections,
    freqz_zpk,
)


ATOL: Final[float] = 1e-6
RTOL: Final[float] = 1e-4


@pytest.fixture
def z() -> Tensor:
    z = torch.tensor(
        [
            [
                0.76 + 0.64j,
                0.69 + 0.71j,
                0.82 + 0.57j,
            ],
            [
                0.64 + 0.76j,
                0.71 + 0.69j,
                0.57 + 0.82j,
            ],
        ]
    )

    return torch.stack([z, z.conj()], dim=-1)


@pytest.fixture
def p() -> Tensor:
    p = torch.tensor(
        [
            [
                0.57 + 0.78j,
                0.85 + 0.48j,
                0.24 + 0.64j,
            ],
            [
                0.78 + 0.57j,
                0.48 + 0.85j,
                0.64 + 0.24j,
            ],
        ]
    )

    return torch.stack([p, p.conj()], dim=-1)


@pytest.fixture
def k() -> Tensor:
    return torch.tensor([0.53, 0.52])


def test_flatten_sections(z: Tensor) -> None:
    z_flat = flatten_sections(z)
    z_flat_baseline = torch.flatten(z, start_dim=-2, end_dim=-1)

    assert z_flat.shape == z_flat_baseline.shape
    assert (z_flat == z_flat_baseline).all()


@pytest.mark.parametrize(
    "N, whole",
    [
        (512, False),
        (512, True),
        (1024, False),
        (1024, True),
    ],
)
def test_freqz_zpk(
    z: Tensor,
    p: Tensor,
    k: Tensor,
    N: int,
    whole: bool,
) -> None:
    z = flatten_sections(z)
    p = flatten_sections(p)

    w, h = freqz_zpk(z, p, k, N=N, whole=whole)

    z = z.numpy()
    p = p.numpy()
    k = k.numpy()

    w = w.numpy()
    h = h.numpy()

    for z, p, k, h in zip(z, p, k, h):
        w_scipy, h_scipy = signal.freqz_zpk(z, p, k, worN=N, whole=whole)

        assert np.allclose(w_scipy, w, rtol=RTOL, atol=ATOL)
        assert np.allclose(h_scipy, h, rtol=RTOL, atol=ATOL)
