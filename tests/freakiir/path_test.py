# SPDX-License-Identifier: MPL-2.0
#
# path_test.py -- path related functionality tests
# Copyright (C) 2025  Jacob Koziej <jacobkoziej@gmail.com>

import pytest

from pathlib import Path
from typing import Optional

from freakiir.path import file_hash


@pytest.mark.parametrize(
    "path, hash, contents",
    (
        # fmt: off
        (Path("foo.txt"), None, None),
        (Path("foo.txt"), None, b''),
        # fmt: on
        (
            Path("foo.txt"),
            "d202d7951df2c4b711ca44b4bcc9d7b363fa4252127e058c1a910ec05b6cd038d71cc21221c031c0359f993e746b07f5965cf8c5c3746a58337ad9ab65278e77",
            b"foo\n",
        ),
    ),
)
def test_file_hash(
    tmp_path: Path,
    path: Path,
    hash: Optional[str],
    contents: Optional[bytes],
) -> None:
    path = tmp_path / path

    if contents is None:
        assert file_hash(path) is None
        return

    with open(path, "wb") as fp:
        fp.write(contents)

    assert file_hash(path) == hash
