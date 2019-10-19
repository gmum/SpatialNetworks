import abc
import dataclasses

import torch


@dataclasses.dataclass
class Base:
    labels: int

    def __post_init__(self):
        self.last: torch.Tensor = None

    def __call__(self, module):
        weight = torch.abs(module.weight)
        if self.last is None:
            self.last = self.first(weight)
        else:
            self.last = self.rest(weight)
        return self.last

    @abc.abstractmethod
    def first(self, weight):
        pass

    @abc.abstractmethod
    def rest(self, weight):
        pass

    @abc.abstractmethod
    def apply(self, weight, mask, task: int):
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
