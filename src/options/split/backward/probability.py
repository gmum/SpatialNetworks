import torch

from . import _base


class Masker(_base.Scale):
    def apply(self, weight, mask, task) -> None:
        probabilities = mask[task] / self.maximum
        weight *= torch.bernoulli(probabilities).reshape(1, -1)
