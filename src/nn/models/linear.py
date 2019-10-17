import torchlayers

from ..layers import SpatialLinear
from ._base import Base


class _Linear(Base):
    @property
    def spatial_type(self):
        return SpatialLinear

    @property
    def regular_type(self):
        return torchlayers.Linear

    def modify_inputs(self, inputs):
        return inputs.reshape(inputs.shape[0], -1)


class SingleOutput(_Linear):
    def create_bottleneck(self, labels, tasks):
        return torchlayers.Linear(labels)


class MultipleOutputs(_Linear):
    def create_bottleneck(self, labels, tasks):
        return torchlayers.Linear(labels * tasks)
