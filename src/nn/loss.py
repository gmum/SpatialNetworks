import dataclasses
import typing

import torch

from .layers import _SpatialConv, _SpatialLinear


def get(args, model):
    if hasattr(args, "where") and args.where is not None:
        return SpatialCrossEntropyLoss(model, args.proximity, args.transport, args.norm)
    return CustomCrossEntropyLoss()


class CustomCrossEntropyLoss:
    def __call__(self, y_pred, y_true):
        # Has to be reshaped for concatenated outputs case
        if len(y_true.shape) > 1:
            y_pred = y_pred.reshape(y_true.shape[0], -1, y_true.shape[1])
        # Make sure it really works this way (though it seems like it)
        return torch.nn.functional.cross_entropy(y_pred, y_true)


class SpatialCrossEntropyLoss:
    def __init__(self, module, proximity, transport, norm):
        self.module = module
        self.proximity = Proximity(proximity)
        self.transport = Transport(transport, norm)

    def __call__(self, y_pred, y_true):
        if len(y_true.shape) > 1:
            y_pred = y_pred.reshape(y_true.shape[0], -1, y_true.shape[1])
        return (
            torch.nn.functional.cross_entropy(y_pred, y_true)
            + self.proximity(self.module)
            + self.transport(self.module)
        )


@dataclasses.dataclass
class Proximity:
    alpha: float
    epsilon: float = 1e-8
    spatial_types: typing.Tuple = (_SpatialLinear, _SpatialConv)

    def __call__(self, module):
        proximity_penalty = []

        for submodule in module.modules():
            if isinstance(submodule, self.spatial_types):
                positions = submodule.positions
                # Get a 2-d array of vectors => [N, N, 2]
                distances = positions.unsqueeze(0) - positions.unsqueeze(1)
                # Calculate squared distances and flatten => [N, N]
                distances = distances.pow(2).sum(-1).view(-1)
                # Take a square root after making sure that there are no zeros
                distances = (distances + self.epsilon).sqrt()
                proximity_penalty.append(torch.exp(-distances).mean().item())

        return self.alpha * torch.tensor(proximity_penalty).mean()


###############################################################################
#
#
#   CHECK WHETHER EINSUM BELOW, MIGHT BE ROOT OF ERRORS. GENERALIZE IF IT'S WRONG
#   DON'T HARDCODE LAYER SHAPES TO BE EQUAL IF NOT NECESSARY.
#
#
###############################################################################


@dataclasses.dataclass
class Transport:
    beta: float
    norm: str
    spatial_types: typing.Tuple = (_SpatialLinear, _SpatialConv)

    def __post_init__(self):
        if self.norm.lower() == "l1":
            self._norm_function = lambda weight: torch.abs(weight)
        elif self.norm.lower() == "l2":
            self._norm_function = lambda weight: torch.pow(weight, 2)
        else:
            raise ValueError("Unsupported weight norm. One of L1/L2 available.")

    def _find_previous_spatial(self, submodules):
        for module in reversed(submodules):
            if isinstance(module, self.spatial_types):
                return module
        return None

    def __call__(self, module):
        transport_penalty = []
        submodules = list(module.modules())
        for i, submodule in enumerate(submodules, start=1):
            if isinstance(submodule, self.spatial_types):
                previous_spatial = self._find_previous_spatial(submodules[:i])
                if previous_spatial is not None:
                    distances = (
                        # Weights are of shape (2, out) for easier generalization
                        # With convolution
                        submodule.positions.T.unsqueeze(1)
                        - previous_spatial.positions.T.unsqueeze(0)
                        # + 1
                    )
                    distances = distances.pow(2).sum(-1).sqrt()
                    normalized_weights = self._norm_function(submodule.weight)
                    #
                    # Check whether this one is truly correct (probably isn't, WIP)
                    #
                    transport_penalty.append(
                        torch.einsum("ij,j...->ij...", distances, normalized_weights)
                        .mean()
                        .item()
                    )

        return self.beta * torch.tensor(transport_penalty).sum()
