import torch

from . import _base


class Masker(_base.Scale):
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
    def apply(self, weight, mask, task) -> None:
        scaling = mask[task] / self.maximum
        weight *= scaling.reshape(1, -1)
