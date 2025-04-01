# SPDX-License-Identifier: MPL-2.0
#
# datamodule.py -- data modules
# Copyright (C) 2025  Jacob Koziej <jacobkoziej@gmail.com>

import torch

from pytorch_lightning.core import LightningDataModule
from torch import (
    Generator,
    Tensor,
)
from torch.utils.data import (
    DataLoader,
    Dataset,
    random_split,
)

from .dataset import (
    ListenHrtf,
    RandomFilterDataset,
    RandomFilterDatasetConfig,
)
from .model import ModelInput


class RandomFilterWithListenHrtf(LightningDataModule):
    def __init__(
        self,
        config: RandomFilterDatasetConfig,
        *,
        batch_size: int = 16,
        num_workers: int = 4,
        split_seed: int = 0x30008AA5
    ) -> None:
        super().__init__()

        self.save_hyperparameters()

    def prepare_data(self) -> None:
        # download & decompress
        _ = ListenHrtf(decompress=True, download=True)

    def setup(self, stage: str) -> None:
        self._train = RandomFilterDataset(self.hparams.config)

        self._val, self._test = random_split(
            ListenHrtf(decompress=False, download=False),
            [0.8, 0.2],
            generator=Generator().manual_seed(self.hparams.split_seed),
        )

    def test_dataloader(self) -> DataLoader:
        return self._dataloader(self._test)

    def train_dataloader(self) -> DataLoader:
        return self._dataloader(self._train)

    def val_dataloader(self) -> DataLoader:
        return self._dataloader(self._val)

    @staticmethod
    def _collate_fn(batch: ModelInput) -> tuple[Tensor, Tensor]:
        w = []
        h = []

        for sample in batch:
            w.append(sample.w)
            h.append(sample.h)

        return torch.stack(w), torch.stack(h)

    def _dataloader(self, dataset: Dataset) -> DataLoader:
        hparams = self.hparams

        num_workers = hparams.num_workers

        if torch.cuda.is_available():
            num_workers *= torch.cuda.device_count()

        return DataLoader(
            dataset,
            batch_size=hparams.batch_size,
            collate_fn=RandomFilterWithListenHrtf._collate_fn,
            num_workers=num_workers,
        )
