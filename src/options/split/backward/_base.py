import abc
import dataclasses

import torch


@dataclasses.dataclass
class Base:
    """Convenience base class defining masking API.

    When called, either runs `first` or `rest` and returns appropriate mask.
    Should be inherited and have `first`, `rest` and `apply` methods implemented.

    Parameters
    ----------
    labels: int
            How many labels were used for each task
    last: Optional[torch.Tensor]
            Last mask created by masker

    """

    labels: int

    def __post_init__(self):
        self.reset()

    def __call__(self, module):
        weight = torch.abs(module.weight.data)
        if self.last is None:
            self.last = self.first(weight)
        else:
            self.last = self.rest(weight)
        return self.last

    def reset(self):
        self.last: torch.Tensor = None

    @abc.abstractmethod
    def first(self, weight):
        """How to create mask during first pass.

        Parameters
        ----------
        weight : torch.Tensor
                Detached weight of specific submodule

        Returns
        -------
        torch.Tensor
                Slicing mask

        """
        pass

    @abc.abstractmethod
    def rest(self, weight):
        """How to create mask during consecutive passes.

        Should be dependent on already existing `self.last` mask.

        Parameters
        ----------
        weight : torch.Tensor
                Detached weight of specific submodule

        Returns
        -------
        torch.Tensor
                Slicing mask

        """
        pass

    @abc.abstractmethod
    def apply(self, weight, mask, task: int) -> None:
        """Apply mask to weight for a given task.

        Side-effect function, should modify `weight` in-place.

        Parameters
        ----------
        weight : torch.Tensor
                Detached weight of specific submodule
        mask: torch.Tensor
                Mask-like object created via `__call__`
        task: int
                Index of current task.

        """
        pass


class Scale(Base):
    def _max(self, mask) -> None:
        maximum, _ = mask.max(dim=0)
        self.maximum = maximum

    def first(self, weight):
        mask = weight.reshape(self.labels, -1, weight.shape[1]).sum(dim=0)
        self._max(mask)
        return mask

    def rest(self, weight):
        mask = torch.stack(
            [weight[self.last == task].sum(dim=0) for task in torch.unique(self.last)]
        )
        self._max(mask)
        return mask

    @abc.abstractmethod
    def apply(self, weight, mask, task: int):
        pass
