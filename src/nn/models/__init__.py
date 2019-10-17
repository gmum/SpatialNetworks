import importlib
import sys

import torch

from . import _dev_utils, convolution, linear


def get(args):
    """Convenience function returning appropriate model based on user arguments.

    Parameters
    ----------
    args: argparse.Namespace
            argparse.ArgumentParser().parse() return value. User provided arguments.

    Returns
    -------
    torch.nn.Module
            Either convolutional or linear model with spatial layers where
            specified. Either single output (for sequential tasks) or with multiple
            outputs (for mixup or concatenation).

    """
    module = args.type.lower()
    if module == "linear":
        module_class = _dev_utils.get(linear, args)
    else:
        module_class = _dev_utils.get(convolution, args)

    network = module_class(
        args.labels,
        len(args.datasets),
        getattr(torch.nn, args.activation),
        args.layers,
        args.where,
    )
    return network
