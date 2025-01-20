# SPDX-License-Identifier: MPL-2.0
#
# dsp.py -- digital signal processing
# Copyright (C) 2025  Jacob Koziej <jacobkoziej@gmail.com>

from einops import rearrange
from torch import Tensor


def flatten_sections(h: Tensor) -> Tensor:
    return rearrange(h, "... sections h -> ... (sections h)")
