# SPDX-License-Identifier: MPL-2.0
#
# dataset.py -- datasets
# Copyright (C) 2025  Jacob Koziej <jacobkoziej@gmail.com>

import builtins

import torch

from dataclasses import dataclass
from typing import Optional

from einops import rearrange
from torch import Tensor
from torch.utils.data import Dataset

from .dsp import (
    construct_sections,
    freqz_zpk,
    order_sections,
)
from .pdf import Pdf


@dataclass
class RandomFilterDatasetConfig:
    sections: int
    pdf_z: Pdf
    pdf_p: Pdf

    all_pass: bool = True
    batch_count: int = 1024
    batch_size: int = 16
    dft_bins: int = 512
    down_order: bool = False
    whole_dft: bool = False

    def __post_init__(self) -> None:
        assert self.sections > 0

        assert self.batch_count > 0
        assert self.batch_size > 0
        assert self.dft_bins > 0


@dataclass
class RandomFilterDatasetOutput:
    w: Tensor
    h: Tensor
    z: Tensor
    p: Tensor
    k: Tensor


class RandomFilterDataset(Dataset):
    def __init__(self, config: RandomFilterDatasetConfig) -> None:
        self.config = config

    def __getitem__(
        self,
        item: int | slice,
    ) -> Optional[RandomFilterDatasetOutput]:
        match type(item):
            case builtins.int:
                if item >= len(self):
                    raise IndexError("dataset index out of range")

                batch_size = 1

            case builtins.slice:
                start = item.start if item.start is not None else 0
                stop = item.stop if item.stop is not None else len(self)
                step = item.step if item.step is not None else 1

                if abs(start) >= len(self):
                    raise IndexError("dataset index out of range")

                if abs(stop) > len(self):
                    raise IndexError("dataset index out of range")

                if step <= 0:
                    raise IndexError("dataset step is not valid")

                start %= len(self)
                stop %= len(self) + 1

                batch_size = round(abs(stop - start) / step)

                if not batch_size:
                    return None

            case _:
                raise TypeError("index must be int or slice")

        config = self.config

        samples = config.sections * batch_size

        p = config.pdf_p(samples)

        if config.all_pass:
            r = 1 / p.abs()
            theta = p.angle()

            z = r * torch.exp(1j * theta)

        else:
            z = config.pdf_z(samples)

        if not isinstance(item, int):
            z = rearrange(z, "(batch z) -> batch z", batch=batch_size)
            p = rearrange(p, "(batch p) -> batch p", batch=batch_size)

        z = order_sections(z, down_order=config.down_order, dim=-1)
        p = order_sections(p, down_order=config.down_order, dim=-1)

        z = construct_sections(z, config.sections, conjugate_pairs=True)
        p = construct_sections(p, config.sections, conjugate_pairs=True)

        k = (
            (torch.norm(p, dim=-1) / torch.norm(z, dim=-1)).squeeze()
            if config.all_pass
            else torch.tensor(1, dtype=z.real.dtype)
        )

        w, h = freqz_zpk(z, p, k, N=config.dft_bins, whole=config.whole_dft)

        return RandomFilterDatasetOutput(w, h, z, p, k)

    def __len__(self) -> int:
        config = self.config

        return config.batch_count * config.batch_size
