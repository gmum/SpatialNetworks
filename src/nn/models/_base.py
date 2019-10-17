import abc
import typing

import torch


class _GetLayer:
    """Return appropriate layer (either spatial or not) according to index.

    Parameters
    ----------
    non_spatial : type
            Class of non-spatial layers
    spatial : type
            Class of spatial layers
    where : Tuple[int]
            Indices where layer type should be spatial. If index is within
            this tuple, layer returned will be spatial

    Returns
    -------
    type
            Either spatial or non-spatial layer type

    """

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
    """Base convenience class for all spatial layers (no matter input shape).

    User has to implement two properties, one function. Optionally, can
    also implement `modify_inputs` in order to modify inputs going into the network
    (used by Linear spatial models to flatten images).

    Please see docs about each function.

    Parameters
    ----------
    labels: int
            How many labels will be used. This information is passed down to
            function creating bottleneck.
    tasks: int
            How many tasks will be provided. This information is passed down to
            function creating bottleneck.
    activation: torch.nn.Module
            Module performing activation after each layer (except final).
    layers: List[int]
            Output sizes of all layers (usually out_channels for convolution
            or out_features for Linear). No input is needed as it's shape will
            be deduced automagically.
    where : List[int]
            List of indices provided by user, pointing which layer should be
            of spatial type (if any).

    """

    def __init__(
        self,
        labels: int,
        tasks: int,
        activation: torch.nn.Module,
        layers: typing.List[int],
        where: typing.List[int],
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
        """Create network bottleneck.

        As bottleneck is network dependent it is up to user inheriting this base
        to specify it.

        As there is no need for input specification, it can be solely
        deduced by number of labels and tasks.

        Parameters
        ----------
        labels: int
                How many labels will be used. This information is passed down to
                function creating bottleneck.
        tasks: int
                How many tasks will be provided. This information is passed down to
                function creating bottleneck.

        """
        pass

    @property
    @abc.abstractmethod
    def regular_type(self):
        """Class providing regular (non-spatial) type.

        For example `torch.nn.Linear` (it's class, not an instance!)
        """
        pass

    @property
    @abc.abstractmethod
    def spatial_type(self):
        """Class providing spatial type.

        For example `nn.layers.SpatialLinear` (it's class, not an instance!)

        You can see available options inside `/src/nn/layers`.
        """
        pass
