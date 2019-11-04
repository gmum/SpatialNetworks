import dataclasses
import typing

import torch

from .layers import spatial


def get(args, model):
    """Based on user input return either spatial cross entropy or regular one.

    SpatialCrossEntropyLoss has additional `transport` and `proximity` constrains;
    see original paper for explanation.

    Parameters
    ----------
    args: argparse.Namespace
            argparse.ArgumentParser().parse() return value. User provided arguments.
    model: torch.nn.Module
            Model whose parameters will be, possibly, used by `transport` and
            `proximity` loss

    Returns
    -------
    typing.Callable
            Loss object (or criterion)

    """
    if hasattr(args, "where") and args.where is not None:
        return SpatialCrossEntropyLoss(model, args.proximity, args.transport, args.regularization, args.norm, args.where, args.labels)
    return CustomCrossEntropyLoss()


class CustomCrossEntropyLoss:
    """Normal CrossEntropyLoss, but reshapes neural net output appropriately.

    For MultiOutput (e.g. mix or concatenation) output of the final layer
    has to be reshaped from `(batch, task * classes)` into `(batch, task, labels)`.

    """

    def __call__(self, y_pred, y_true):
        # Has to be reshaped for concatenated outputs case
        if len(y_true.shape) > 1:
            y_pred = y_pred.reshape(y_true.shape[0], -1, y_true.shape[1])
        # Make sure it really works this way (though it seems like it)
        return torch.nn.functional.cross_entropy(y_pred, y_true)


class SpatialCrossEntropyLoss:
    """Normal CrossEntropyLoss, but reshapes neural net output appropriately.

    For MultiOutput (e.g. mix or concatenation) output of the final layer
    has to be reshaped from `(batch, task * classes)` into `(batch, task, labels)`.

    Parameters
    ----------
    module: torch.nn.Module
            Module whose weights will be constrained by proximity and transport loss.
    proximity: float
            Proximity hyperparameter
    transport: float
            Transport hyperparameter
    norm: str
            Either "l1" or "l2" (case insensitive), representing LP norm to be used.

    """

    def __init__(
            self, module, proximity, transport, regularization,
            norm, where, labels
        ):
        self.module = module
        self.labels = labels
        self.proximity = Proximity(proximity, where)
        self.transport = Transport(transport, norm, where)
        self.regularization = Regularization(regularization, norm)

    def __call__(self, y_pred, y_true):
        if len(y_true.shape) > 1:
            y_pred = y_pred.reshape(y_true.shape[0], -1, y_true.shape[1])
        else: # Sequential setting
            indices = (y_true // self.labels).long()
            indices = indices.repeat_interleave(self.labels)
            y_pred = y_pred.reshape(y_true.shape[0], self.labels, -1)
            y_pred = y_pred.gather(2, indices.view(-1, self.labels, 1)).squeeze()
            y_true %= self.labels

        return (
            torch.nn.functional.cross_entropy(y_pred, y_true)
            + self.proximity(self.module)
            + self.transport(self.module)
            + self.regularization(self.module)
        )

@dataclasses.dataclass
class Regularization:
    gamma: float
    norm: str

    def __call__(self, module):
        reg_penalty = []

        for submodule in module.modules():
            if hasattr(submodule, "weight"):
                if self.norm == "l1":
                    reg_penalty += [torch.sum(torch.abs(submodule.weight))]
                elif self.norm == "l2":
                    reg_penalty += [torch.sum(torch.pow(submodule.weight, 2))]
                else:
                    raise ValueError

        return self.gamma * torch.stack(reg_penalty).sum()

@dataclasses.dataclass
class Proximity:
    """Regularization term discouraging spatial neurons within layer from being too close.

    Like in biological network, neurons have some physical dimensions, hence
    they cannot be "too close".

    Parameters
    ----------
    alpha: float
            Hyperparameter regarding strength of proximity penalty. The higher,
            the more penalty is incurred upon network for grouping neurons together.
    epsilon: float, optional
            Small non-zero value in rare case distance is too small to
            have `sqrt` taken from it.
    spatial_types: Tuple[type]
            Tuple containing types to be considered spatial.
            Default: (SpatialLinear, SpatialConv)

    """

    alpha: float
    where: typing.List[int]
    epsilon: float = 1e-8

    def __call__(self, module):
        proximity_penalty = []

        spatial_idx = 0
        for submodule in module.modules():
            if spatial(submodule):
                if spatial_idx in self.where:
                    positions = submodule.positions.T
                    # Get a 2-d array of vectors => [N, N, 2]
                    distances = positions.unsqueeze(0) - positions.unsqueeze(1)
                    # Calculate squared distances and flatten => [N, N]
                    distances = distances.pow(2).sum(-1).view(-1)
                    # Take a square root after making sure that there are no zeros
                    distances = (distances + self.epsilon).sqrt()
                    proximity_penalty.append(torch.exp(-distances).mean())
                spatial_idx += 1

        return self.alpha * torch.stack(proximity_penalty).mean()


@dataclasses.dataclass
class Transport:
    """Regularization term discouraging long connections between layers.

    If the connection spatial parameter is large, network should drive
    it's corresponding network to smaller value in order to improve
    overall loss induced by this penalty.

    Parameters
    ----------
    beta: float
            Hyperparameter regarding strength of proximity penalty. The higher,
            the more penalty is incurred upon network for grouping neurons together.
    norm: str
            Norm to be used for distance calculation. Either "l1" or "l2" case
            insensitive allowed.
    spatial_types: Tuple[type], optional
            Tuple containing types to be considered spatial.
            Default: (SpatialLinear, SpatialConv)

    """

    beta: float
    norm: str
    where: typing.List[int]

    def __post_init__(self):
        if self.norm.lower() == "l1":
            self._norm_function = lambda weight: torch.abs(weight)
        elif self.norm.lower() == "l2":
            self._norm_function = lambda weight: torch.pow(weight, 2)
        else:
            raise ValueError("Unsupported weight norm. One of L1/L2 available.")

    def _find_previous_spatial(self, submodules):
        for module in reversed(submodules):
            if spatial(module):
                return module
        return None

    def __call__(self, module):
        transport_penalty = []
        submodules = list(module.modules())
        spatial_idx = 0
        for i, submodule in enumerate(submodules):
            if spatial(submodule):
                previous_spatial = self._find_previous_spatial(submodules[:i])
                if previous_spatial is not None and spatial_idx in self.where:
                    distances = (
                        # Weights are of shape (2, out) for easier generalization
                        # With convolution
                        submodule.positions.T.unsqueeze(1)
                        - previous_spatial.positions.T.unsqueeze(0)
                        # + 1
                    )
                    distances = distances.pow(2).sum(-1).sqrt()
                    normalized_weights = self._norm_function(submodule.weight)

                    transport_penalty.append(
                        (distances * normalized_weights).mean()
                    )
                spatial_idx += 1

        return self.beta * torch.stack(transport_penalty).mean()
