# SPDX-License-Identifier: MPL-2.0
#
# callbacks.py -- callbacks
# Copyright (C) 2025  Jacob Koziej <jacobkoziej@gmail.com>

from typing import Any

from hydra.experimental.callback import Callback
from omegaconf import (
    DictConfig,
    open_dict,
)

from .._version import __version__


class Version(Callback):
    def on_run_start(
        self,
        config: DictConfig,
        **kwargs: tuple[str, Any],
    ) -> None:
        with open_dict(config):
            config.version = __version__
