# SPDX-License-Identifier: MPL-2.0
#
# dsp_test.py -- digital signal processing tests
# Copyright (C) 2025  Jacob Koziej <jacobkoziej@gmail.com>

import torch
import pytest

from torch import Tensor

from freakiir.dsp import flatten_sections


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
