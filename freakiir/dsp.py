# SPDX-License-Identifier: MPL-2.0
#
# dsp.py -- digital signal processing
# Copyright (C) 2025  Jacob Koziej <jacobkoziej@gmail.com>

import torch

from typing import Optional

from einops import rearrange
from torch import (
    Tensor,
    pi,
)


def flatten_sections(h: Tensor) -> Tensor:
    return rearrange(h, "... sections h -> ... (sections h)")


def freqz_zpk(
    z: Tensor,
    p: Tensor,
    k: Tensor,
    *,
    N: int = 512,
    whole: bool = False,
) -> tuple[Tensor, Tensor]:
    dtype = z.dtype
    assert p.dtype == dtype

    device = z.device
    assert p.device == device
    assert k.device == device

    def polyvalfromroots(x: Tensor, r: Tensor) -> Tensor:
        r = r.reshape(r.shape + (1,) * x.ndim)

        return torch.prod(x - r, -2)

    k = k.reshape(k.shape + (1,) * (z.ndim - 1))

    end = (2 if whole else 1) * pi

    # our interval is right-open
    end -= end / N

    w = torch.linspace(0, end, N).to(device)
    h = torch.exp(1j * w)

    h = k * polyvalfromroots(h, z) / polyvalfromroots(h, p)

    return w, h


def order_sections(
    x: Tensor,
    *,
    down_order: bool = False,
    dim: int = -1,
) -> Tensor:
    indices = torch.argsort(x.abs(), descending=down_order, dim=dim)

    return torch.take_along_dim(x, indices, dim=dim)


def unwrap(
    x: Tensor,
    *,
    discontinuity: Optional[float] = None,
    period: float = 2 * pi,
    axis: int = -1,
) -> Tensor:
    if discontinuity is None:
        discontinuity = period / 2

    high: float = period / 2
    low: float = -high

    correction_slice = [slice(None, None)] * x.ndim

    correction_slice[axis] = slice(1, None)

    correction_slice = tuple(correction_slice)

    dd = torch.diff(x, axis=axis)
    ddmod = torch.remainder(dd - low, period) + low

    ph_correct = ddmod - dd
    ph_correct = torch.where(dd.abs() < discontinuity, 0, ph_correct)

    x[correction_slice] = x[correction_slice] + ph_correct.cumsum(axis)

    return x
