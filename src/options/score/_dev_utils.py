import dataclasses
import pathlib
import typing

import torch
import torchfunc

import nn


def get_models(folder: pathlib.Path):
    """Obtain per-task models from `folder`.

    Yields
    ------
    torch.nn.Module
        Consecutive splitted models

    """
    for path in pathlib.Path(folder).glob("*"):
        yield torch.load(path)


def run(loop, gatherer) -> None:
    """Run validation loop and print gathered results."""
    for result in loop():
        gatherer(result)
    print(
        "=================================RESULTS======================================"
    )
    nn.metrics.print_results(gatherer.get())
    print(
        "===================================END========================================"
    )
