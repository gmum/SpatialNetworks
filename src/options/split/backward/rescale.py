import torch

from . import _base


class Masker(_base.Scale):
    def apply(self, weight, mask, task) -> None:
        scaling = mask[task] / self.maximum
        weight *= scaling.reshape(1, -1)
