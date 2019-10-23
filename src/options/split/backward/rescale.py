import torch

from . import _base


class Masker(_base.Base):
    """For each task, each neuron will be rescaled based on it's task input.

    The more connected neuron to the task, the smaller the divisor will be.
    Neurons having highest weight connected to specific task will not
    be rescaled at all.

    Parameters
    ----------
    labels: int
            How many labels were used for each task
    last: Optional[torch.Tensor]
            Last mask created by masker

    """

    @staticmethod
    def _create_mask(label_summed):
        maximum, _ = label_summed.max(dim=0)
        return torch.bernoulli(1 - (summed / maximum.unsqueeze(0))).bool()

    def first(self, weight):
        return Masker._create_mask(
            weight.reshape(self.labels, -1, weight.shape[1]).sum(dim=0)
        )

    def rest(self, weight):
        return Masker._create_mask(
            torch.stack(
                [
                    weight[self.last[task]].sum(dim=0)
                    for task in range(self.last.shape[0])
                ]
            )
        )

    def apply(self, weight, mask, task) -> None:
        weight *= mask[task].unsqueeze(0)
