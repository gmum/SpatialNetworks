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
    """Convolution network with single output (for sequential input)."""

    def create_bottleneck(self, labels, _, linear_cls):
        return torch.nn.Sequential(
            torchlayers.GlobalMaxPool(), linear_cls(labels)
        )


class MultipleOutputs(Conv):
    """Convolution network with multiple outputs (for concatenation or mix input)."""

    def create_bottleneck(self, labels, tasks, linear_cls):
        return torch.nn.Sequential(
            torchlayers.GlobalMaxPool(), linear_cls(labels * tasks)
        )
