import torch
import torchfunc

import data

from .._dev_utils import get_datasets
from . import _dev_utils, finalize, recorders, reduction


def run(args):
    if args.input.lower() == "sequential":
        datasets = get_datasets(args, args.train)
        print(f"Datasets used: {[type(dataset).__name__ for dataset in datasets]}\n")
        dataset = data.datasets.get(args, *datasets)
        model = _dev_utils.get_model(args)
        print(f"Model used for recording:\n{model}\n")

        activation_recorders = recorders.get(dataset, reduction.get(args))
        recorders.register(model, activation_recorders)
        print(f"Recording activations...\n")
        _dev_utils.record_state(model, dataset, args)
        finalize.apply(activation_recorders, args)
        print(f"Saving recorded data at: {args.save}\n")
        recorders.save(activation_recorders, args)

    else:
        print(
            "For now only sequential task supported for activation based recordings, passing..."
        )
