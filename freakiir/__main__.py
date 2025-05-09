# SPDX-License-Identifier: MPL-2.0
#
# __main__.py -- main
# Copyright (C) 2025  Jacob Koziej <jacobkoziej@gmail.com>

import sys

import hydra

from argparse import ArgumentParser
from typing import Any

from hydra.utils import instantiate
from omegaconf import DictConfig

from freakiir.pdf import Pdf


def _pdf(*args: list[Any], **kwargs: dict[str, Any]) -> Pdf:
    pdfs = []

    pdfs += list(filter(callable, args))
    pdfs += list(filter(callable, kwargs.values()))

    return pdfs


@hydra.main(
    config_name="config",
    config_path="conf",
    version_base="1.2",
)
def _train(cfg: DictConfig) -> None:
    datamodule = instantiate(cfg.datamodule)
    model = instantiate(cfg.model)
    trainer = instantiate(cfg.trainer)

    trainer.fit(
        model,
        ckpt_path=cfg.checkpoint.resume,
        datamodule=datamodule,
    )


def main() -> None:
    parser = ArgumentParser(
        description="Frequency Response Enhanced by All-pass estimations of Kth order Infinite Impulse Responses",
    )
    subparsers = parser.add_subparsers(
        dest="sub_command",
    )
    _ = subparsers.add_parser("train")

    args, unknownargs = parser.parse_known_args()

    sub_command = args.sub_command or "train"

    match sub_command:
        case "train":
            # hydra doesn't let us override its argument vector
            sys.argv = sys.argv[:1] + unknownargs

            _train()


if __name__ == "__main__":
    sys.exit(main())
