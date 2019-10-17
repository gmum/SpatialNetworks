import importlib
import sys

import torch

from . import _dev_utils, convolution, linear


def get(args):
    # Make module matching with string
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
