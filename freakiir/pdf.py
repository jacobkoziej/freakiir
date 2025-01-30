# SPDX-License-Identifier: MPL-2.0
#
# pdf.py -- probability density functions
# Copyright (C) 2025  Jacob Koziej <jacobkoziej@gmail.com>

from collections.abc import Callable
from typing import NewType

from torch import Tensor


Pdf = NewType("Pdf", Callable[[int], Tensor])
