# SPDX-License-Identifier: MPL-2.0
#
# path.py -- path related functionality
# Copyright (C) 2025  Jacob Koziej <jacobkoziej@gmail.com>

from pathlib import Path
from typing import Final

from platformdirs import user_cache_dir


_APP_AUTHOR: Final[str] = "jacobkoziej"
_APP_NAME: Final[str] = "freakiir"


def cache_directory() -> Path:
    return Path(user_cache_dir(_APP_NAME, _APP_AUTHOR, ensure_exists=True))


def datasets_directory() -> Path:
    directory = cache_directory() / "datasets"

    directory.mkdir(parents=True, exist_ok=True)

    return directory
