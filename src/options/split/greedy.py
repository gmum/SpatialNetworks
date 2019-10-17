import dataclasses
import pathlib

import torch

from . import _dev_utils


@dataclasses.dataclass
class MaskCreator:
    labels: int

    def __call__(self, module):
        weight = torch.abs(module.weight)
        return weight.reshape(self.labels, -1, weight.shape[1]).sum(dim=0).argmax(dim=0)


def apply_mask(module, mask, task):
    module.weight.data[:, mask != task] *= 0
