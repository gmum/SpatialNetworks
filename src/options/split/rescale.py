import dataclasses
import pathlib

import torch

from . import _dev_utils


@dataclasses.dataclass
class MaskCreator:
    labels: int

    def __call__(self, module):
        weight = torch.abs(module.weight)
        return weight.reshape(self.labels, -1, weight.shape[1]).sum(dim=0)


def apply_mask(module, mask, task):
    maximum, _ = mask.max(dim=0)
    mask = mask[task] / maximum
    module.weight.data *= mask.reshape(1, -1)
