import abc
import typing

import torch


class _GetLayer:
    def __init__(self, non_spatial, spatial, where):
        self.non_spatial = non_spatial
        self.spatial = spatial
        self.where = where

    def __call__(self, index):
        if self.where is None:
            return self.non_spatial
        if index in self.where:
            return self.spatial
        return self.non_spatial


class Base(torch.nn.Module):
    def __init__(
        self,
        labels: int,
        tasks: int,
        activation: torch.nn.Module,
        layers: typing.List[int],
        where,
    ):
        super().__init__()
        layer_type_getter = _GetLayer(self.regular_type, self.spatial_type, where)

        if layers is None:
            layers = []

        self.layers = torch.nn.Sequential(
            *[layer_type_getter(i)(width) for i, width in enumerate(layers)]
        )
        self.bottleneck = self.create_bottleneck(labels, tasks)

    def forward(self, inputs):
        return self.bottleneck(self.layers(self.modify_inputs(inputs)))

    # Override to modify [batch, channels, width, height] input in submodule
    def modify_inputs(self, inputs):
        return inputs

    @abc.abstractmethod
    def create_bottleneck(self, labels, tasks):
        pass

    @property
    @abc.abstractmethod
    def regular_type(self):
        pass

    @property
    @abc.abstractmethod
    def non_spatial_type(self):
        pass
