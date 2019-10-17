"""
This module provides functors performing single pass through dataset.
Those should be used in some kind of looping structure if one wants to use
multiple epochs.

"""

import dataclasses
import typing

import torch
import torchfunc


# Single pass through data sample
@dataclasses.dataclass
class Train:
    """Perform standard training on provided sample.

    Parameters
    ----------
    module: torch.nn.Module
            Module to be trained.
    criterion: typing.Callable
            Criterion (e.g. `torch.nn.CrossEntropy`)
    optimizer: torch.optim.Optimizer
            Instance of PyTorch optimizer with `step` and `zero_grad`
            method implemented.
    cuda: bool
            Whether to use cuda for training.

    """

    module: torch.nn.Module
    criterion: typing.Callable
    optimizer: torch.optim.Optimizer
    cuda: bool

    def __post_init__(self):
        if self.cuda:
            self.module = self.module.cuda()

    def __call__(self, sample, *_):
        self.module.train()
        self.optimizer.zero_grad()
        image, label = sample
        if self.cuda:
            image = image.cuda()
            label = label.cuda()
        predicted = self.module(image)
        loss = self.criterion(predicted, label)
        loss.backward()
        self.optimizer.step()
        return (loss, predicted, label)


# Single pass through data sample
@dataclasses.dataclass
class Validation:
    """Perform standard validation on provided sample.

    Module will be put in `eval` mode. User should use `train` method
    if he want to train it further.

    Parameters
    ----------
    module: torch.nn.Module
            Module to be trained.
    criterion: typing.Callable
            Criterion (e.g. `torch.nn.CrossEntropy`)
    cuda: bool
            Whether to use cuda for training.

    """

    module: torch.nn.Module
    criterion: typing.Callable
    cuda: bool

    def __post_init__(self):
        if self.cuda:
            self.module = self.module.cuda()

    def __call__(self, sample, *_):
        self.module.eval()
        image, label = sample
        if self.cuda:
            image = image.cuda()
            label = label.cuda()
        predicted = self.module(image)
        loss = self.criterion(predicted, label)
        return (loss, predicted, label)
