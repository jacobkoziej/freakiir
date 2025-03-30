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


def construct_sections(
    h: Tensor,
    sections: int,
    *,
    conjugate_pairs: bool = False,
) -> Tensor:
    h = rearrange(
        h,
        "... (sections h) -> ... sections h",
        sections=sections,
        h=1 if conjugate_pairs else 2,
    )

    if conjugate_pairs:
        h = torch.cat([h, h.conj()], dim=-1)

    return h


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

    k = k.reshape(k.shape + (1,) * (z.ndim - 1))

    end = (2 if whole else 1) * pi

    # our interval is right-open
    end -= end / N

    w = torch.linspace(0, end, N).to(device)
    h = torch.exp(1j * w)

    h = k * polyvalfromroots(h, z) / polyvalfromroots(h, p)

    return w, h


def order_sections(
    h: Tensor,
    *,
    down_order: bool = False,
    dim: int = -1,
) -> Tensor:
    indices = torch.argsort(h.abs(), descending=down_order, dim=dim)

    return torch.take_along_dim(h, indices, dim=dim)


def polyvalfromroots(x: Tensor, r: Tensor) -> Tensor:
    r = r.reshape(r.shape + (1,) * x.ndim)

    return torch.prod(x - r, -2)


def unwrap(
    h: Tensor,
    *,
    discontinuity: Optional[float] = None,
    period: float = 2 * pi,
    dim: int = -1,
) -> Tensor:
    if discontinuity is None:
        discontinuity = period / 2

    high: float = period / 2
    low: float = -high

    correction_slice = [slice(None, None)] * h.ndim

    correction_slice[dim] = slice(1, None)

    correction_slice = tuple(correction_slice)

    dd = torch.diff(h, dim=dim)
    ddmod = torch.remainder(dd - low, period) + low

    ph_correct = ddmod - dd
    ph_correct = torch.where(dd.abs() < discontinuity, 0, ph_correct)

    h[correction_slice] = h[correction_slice] + ph_correct.cumsum(dim)

    return h
