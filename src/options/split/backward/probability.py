import torch

from . import _base


class Masker(_base.Scale):
    """For each task, each neuron has probability to be kept.

    The more connected neuron to task, the higher the probability to keep it.
    Other neurons will be zeroed-out

    Parameters
    ----------
    labels: int
            How many labels were used for each task
    last: Optional[torch.Tensor]
            Last mask created by masker

    """

    def apply(self, weight, mask, task) -> None:
        probabilities = mask[task] / self.maximum
        weight *= torch.bernoulli(probabilities).reshape(1, -1)
