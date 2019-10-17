import torch

import torchlayers

from ..layers import SpatialConv
from ._base import Base


class Conv(Base):
    @property
    def spatial_type(self):
        return SpatialConv

    @property
    def regular_type(self):
        return torchlayers.Conv2d


class SingleOutput(Conv):
    def create_bottleneck(self, labels, _):
        return torch.nn.Sequential(
            torchlayers.GlobalMaxPool(), torchlayers.Linear(labels)
        )


class MultipleOutputs(Conv):
    def create_bottleneck(self, labels, tasks):
        return torch.nn.Sequential(
            torchlayers.GlobalMaxPool(), torchlayers.Linear(labels * tasks)
        )
