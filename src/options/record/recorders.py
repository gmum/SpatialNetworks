import functools
import pathlib
import typing

import torchfunc

import nn
import torchlayers

RECORDER_TYPES = (
    torchlayers.Linear,
    torchlayers.Conv,
    nn.layers._SpatialConv,
    nn.layers._SpatialLinear,
)


def get(dataset, reduction) -> typing.List:
    def condition(index):
        return dataset.dataset == index

    return [
        torchfunc.hooks.recorders.ForwardPre(
            condition=functools.partial(condition, i), reduction=reduction
        )
        for i in range(dataset.inner_datasets)
    ]


def register(model, recorders) -> None:
    for task_recorder in recorders:
        task_recorder.modules(model, types=RECORDER_TYPES)


def save(recorders, args) -> None:
    path = pathlib.Path(args.save)
    for index, task_recorder in enumerate(recorders):
        task_recorder.save(path / str(index), mkdir=True, parents=True, exist_ok=True)
