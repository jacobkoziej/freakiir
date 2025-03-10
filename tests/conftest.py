# SPDX-License-Identifier: MPL-2.0
#
# conftest.py -- top-level conftest
# Copyright (C) 2025  Jacob Koziej <jacobkoziej@gmail.com>

import pytest

from torch import Generator


@pytest.fixture
def generator(seed) -> Generator:
    return Generator().manual_seed(seed)


@pytest.fixture(scope="session")
def samples() -> int:
    return 1 << 12


@pytest.fixture(scope="session")
def seed() -> int:
    return 0xF14CE2E4
