import json

import torchfunc
from torch.utils.tensorboard import SummaryWriter

import data
import nn

from .._dev_utils import get_datasets
from . import _dev_utils


def run(args):
    # Seed and load parameters
    hyperparams = json.load(open(args.hyperparams))
    torchfunc.seed(hyperparams["seed"], cuda=True)

    # Setup appropriate model and data
    # Datasets specified by user
    datasets = get_datasets(args, train=args.train)
    print(f"Datasets used: {[type(dataset).__name__ for dataset in datasets]}\n")

    dataset = data.datasets.get(args, *datasets)
    models = _dev_utils.get_models(args.models)
    for task, model in enumerate(models):
        print(
            f"==================================TASK {task}======================================"
        )
        print(f"Model:\n {model}")

        # Model is needed to register it's params for transport & proximity loss functions
        loss = nn.loss.get(args, model)

        single_pass = nn.passes.Validation(model, loss, args.cuda)
        writer = SummaryWriter(log_dir=args.tensorboard)
        loop = nn.train.get_loop(single_pass, dataset, hyperparams)
        gatherer = nn.metrics.get(
            writer, dataset, stage=f"Task{task}", tasks=len(datasets)
        )

        # Run training and validation
        _dev_utils.run(loop, gatherer)
        print("\n")
