import torch

from . import _base


class Masker(_base.Base):
    """Split networks greedily as shown in original work.

    See: `https://arxiv.org/abs/1910.02776`__ for more details.

    Parameters
    ----------
    labels: int
            How many labels were used for each task
    last: Optional[torch.Tensor]
            Last mask created by masker

    """

    def first(self, weight):
        return weight.reshape(self.labels, -1, weight.shape[1]).sum(dim=0).argmax(dim=0)

    def rest(self, weight):
        return torch.stack(
            [weight[self.last == task].sum(dim=0) for task in torch.unique(self.last)]
        ).argmax(dim=0)

    def apply(self, weight, mask, task) -> None:
        weight[mask != task, :] *= 0
