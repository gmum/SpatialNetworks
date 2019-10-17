import dataclasses
import typing

import torch
import torchfunc


# Single pass through data sample
@dataclasses.dataclass
class Train:
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
