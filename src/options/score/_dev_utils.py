import dataclasses
import pathlib
import typing

import torch
import torchfunc

import nn


def get_models(folder: pathlib.Path):
    for path in pathlib.Path(folder).glob("*"):
        yield torch.load(path)


def run(loop, gatherer):
    for result in loop():
        gatherer(result)
    print(
        "=================================RESULTS======================================"
    )
    nn.metrics.print_results(gatherer.get())
    print(
        "===================================END========================================"
    )


# Single pass through data sample
@dataclasses.dataclass
class ValidationPass:
    module: torch.nn.Module
    criterion: typing.Callable
    optimizer: torch.optim.Optimizer
    cuda: bool

    def __post_init__(self):
        if self.cuda:
            self.module = self.module.cuda()
        self.module.eval()
        torchfunc.module.freeze(self.module)

    def __call__(self, sample, *_):
        image, label = sample
        if self.cuda:
            image = image.cuda()
            label = label.cuda()
        predicted = self.module(image)
        loss = self.criterion(predicted, label)
        return (loss, predicted, label)
