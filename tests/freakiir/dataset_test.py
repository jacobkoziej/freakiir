# SPDX-License-Identifier: MPL-2.0
#
# dataset_test.py -- datasets tests
# Copyright (C) 2025  Jacob Koziej <jacobkoziej@gmail.com>

import pytest
import torch

from pytest import FixtureRequest
from torch import (
    Tensor,
    pi,
)

from freakiir.dataset import (
    RandomFilterDataset,
    RandomFilterDatasetConfig,
)
from freakiir.dsp import (
    freqz_zpk,
    order_sections,
)


class TestRandomFilterDatasetConfig:
    @pytest.fixture(params=[-1, 0, 1])
    def sections(self, request: FixtureRequest) -> int:
        return request.param

    @pytest.fixture(params=[-1, 0, 1])
    def batch_count(self, request: FixtureRequest) -> int:
        return request.param

    @pytest.fixture(params=[-1, 0, 1])
    def batch_size(self, request: FixtureRequest) -> int:
        return request.param

    @pytest.fixture(params=[-1, 0, 1])
    def dft_bins(self, request: FixtureRequest) -> int:
        return request.param

    def test_post_init_assert(
        self,
        sections: int,
        batch_count: int,
        batch_size: int,
        dft_bins: int,
    ) -> None:
        args = {
            "sections": sections,
            "batch_count": batch_count,
            "batch_size": batch_size,
            "dft_bins": dft_bins,
        }

        success = True

        for val in args.values():
            success &= val > 0

        args["pdf_z"] = None
        args["pdf_p"] = None

        if success:
            _ = RandomFilterDatasetConfig(**args)
            return

        with pytest.raises(AssertionError):
            _ = RandomFilterDatasetConfig(**args)


class TestRandomFilterDataset:
    @pytest.fixture(params=[1, 2, 3, 4])
    def sections(self, request: FixtureRequest) -> int:
        return request.param

    @pytest.fixture(params=[False, True])
    def all_pass(self, request: FixtureRequest) -> bool:
        return request.param

    @pytest.fixture(params=[1024, 3])
    def batch_count(self, request: FixtureRequest) -> int:
        return request.param

    @pytest.fixture(params=[16, 7])
    def batch_size(self, request: FixtureRequest) -> int:
        return request.param

    @pytest.fixture(params=[512, 43])
    def dft_bins(self, request: FixtureRequest) -> int:
        return request.param

    @pytest.fixture(params=[False, True])
    def down_order(self, request: FixtureRequest) -> bool:
        return request.param

    @pytest.fixture(params=[False, True])
    def whole_dft(self, request: FixtureRequest) -> bool:
        return request.param

    @pytest.fixture(autouse=True)
    def config(self) -> None:
        self.config = RandomFilterDatasetConfig(
            sections=1,
            pdf_z=TestRandomFilterDataset.pdf_z,
            pdf_p=TestRandomFilterDataset.pdf_p,
        )

    @staticmethod
    def pdf(samples: int, phase_offset: float) -> Tensor:
        theta = torch.linspace(0, pi, samples)
        r = torch.linspace(0.1, 0.9, samples)

        return r * torch.exp(1j * (theta + phase_offset))

    @staticmethod
    def pdf_z(samples: int) -> Tensor:
        return TestRandomFilterDataset.pdf(samples, 0.0)

    @staticmethod
    def pdf_p(samples: int) -> Tensor:
        return TestRandomFilterDataset.pdf(samples, -pi / 2)

    def test_getitem_all_pass(self, all_pass: bool) -> None:
        config = self.config

        config.all_pass = all_pass

        dataset = RandomFilterDataset(config)

        output = dataset[0]
        h = output.h

        allclose = torch.allclose(h.abs(), torch.tensor(1, dtype=h.real.dtype))

        assert allclose if all_pass else (not allclose)

    def test_getitem_dft(self, dft_bins: int, whole_dft: bool) -> None:
        config = self.config

        config.dft_bins = dft_bins
        config.whole_dft = whole_dft

        dataset = RandomFilterDataset(config)

        output = dataset[0]

        w, h = freqz_zpk(
            output.z,
            output.p,
            output.k,
            N=dft_bins,
            whole=whole_dft,
        )

        assert torch.allclose(output.w, w)
        assert torch.allclose(output.h, h)

    def test_getitem_down_order(self, down_order: bool) -> None:
        config = self.config

        config.down_order = down_order

        dataset = RandomFilterDataset(config)

        output = dataset[0]

        z = order_sections(output.z, down_order=down_order)
        p = order_sections(output.p, down_order=down_order)

        assert torch.allclose(output.z, z)
        assert torch.allclose(output.p, p)

    @pytest.mark.parametrize(
        "item, items",
        [
            (0, 1),
            (slice(0, 0, None), 0),
            (slice(0, 15, None), 15),
            (slice(0, 15, 2), 8),
            (slice(-1, None, None), 1),
            (slice(-8, -1, 3), 3),
        ],
    )
    def test_getitem_item_count(
        self,
        item: int | slice,
        items: int,
        dft_bins: int,
    ) -> None:
        config = self.config

        config.dft_bins = dft_bins

        dataset = RandomFilterDataset(config)

        output = dataset[item]

        if not items:
            assert output is None
            return

        shape = (items,) * (not isinstance(item, int)) + (config.sections,)

        assert output.w.shape == (dft_bins,)
        assert output.h.shape == shape + (dft_bins,)
        assert output.z.shape == shape + (2,)
        assert output.p.shape == shape + (2,)
        assert output.k.shape == (items,) * (items > 1)

    def test_len(self, batch_count: int, batch_size: int) -> None:
        config = self.config

        config.batch_count = batch_count
        config.batch_size = batch_size

        dataset = RandomFilterDataset(config)

        assert len(dataset) == batch_count * batch_size
