# SPDX-License-Identifier: MPL-2.0
#
# dsp.py -- digital signal processing
# Copyright (C) 2025  Jacob Koziej <jacobkoziej@gmail.com>

import torch

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
