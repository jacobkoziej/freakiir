# SPDX-License-Identifier: MPL-2.0
#
# path.py -- path related functionality
# Copyright (C) 2025  Jacob Koziej <jacobkoziej@gmail.com>

from hashlib import blake2b
from mmap import (
    ACCESS_READ,
    mmap,
)
from pathlib import Path
from typing import (
    Final,
    Optional,
)

from platformdirs import user_cache_dir


_APP_AUTHOR: Final[str] = "jacobkoziej"
_APP_NAME: Final[str] = "freakiir"


def cache_directory() -> Path:
    return Path(user_cache_dir(_APP_NAME, _APP_AUTHOR, ensure_exists=True))


def datasets_directory() -> Path:
    directory = cache_directory() / "datasets"

    directory.mkdir(parents=True, exist_ok=True)

    return directory


def file_hash(path: Path | str) -> Optional[str]:
    if isinstance(path, str):
        path = Path(path)

    if not path.is_file():
        return None

    if not path.stat().st_size:
        return None

    with open(path) as fp, mmap(fp.fileno(), 0, access=ACCESS_READ) as fp:
        return blake2b(fp).hexdigest()
