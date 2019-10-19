import torch

from . import _base


class Masker(_base.Base):
    def first(self, weight):
        return weight.reshape(self.labels, -1, weight.shape[1]).sum(dim=0).argmax(dim=0)

    def rest(self, weight):
        return torch.stack(
            [weight[self.last == task].sum(dim=0) for task in torch.unique(self.last)]
        ).argmax(dim=0)

    def apply(self, weight, mask, task) -> None:
        weight[:, mask != task] *= 0
