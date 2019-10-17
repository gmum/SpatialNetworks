import dataclasses
import typing

import torch
import tqdm


# Single pass through data sample
@dataclasses.dataclass
class Pass:
    module: torch.nn.Module
    criterion: typing.Callable
    optimizer: torch.optim.Optimizer
    cuda: bool

    def __post_init__(self):
        if self.cuda:
            self.module = self.module.cuda()

    def __call__(self, sample, *_):
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


# Experimental idea to wrap criterion and optimizer as single update
# @dataclasses.dataclass
# class Update:
#     criterion: typing.Callable
#     optimizer: torch.optim.Optimizer
#     clear: bool = True
#     backward: bool = True
#     step: bool = True

#     def __enter__(self):
#         if self.clear:
#             self.optimizer.zero_grad()
#         return self

#     def __call__(self, *args, **kwargs):
#         loss = self.criterion(*args, **kwargs)
#         if self.backward:
#             loss.backward()
#         return loss

#     def __exit__(self, *_):
#         if self.step:
#             self.optimizer.step()
