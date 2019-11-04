import json
import operator
import pathlib

import torch
import torchfunc
from torch.utils.tensorboard import SummaryWriter  # well,

import data
import nn

from .._dev_utils import get_datasets
from . import _dev_utils


def run(args):
    # Seed and load parameters
    hyperparams = json.load(open(args.hyperparams))
    torchfunc.seed(hyperparams["seed"], cuda=True)

    # Setup appropriate model and data
    model = nn.models.get(args)
    if args.load:
        torch.load(args.load)

    # Datasets specified by user
    train_datasets = get_datasets(args, train=True)
    validation_datasets = get_datasets(args, train=False)
    print(f"Datasets used: {[type(dataset).__name__ for dataset in train_datasets]}\n")

    # Appropriate dataloader constructed from those datasets
    train, validation = (
        data.datasets.get(args, *train_datasets),
        data.datasets.get(args, *validation_datasets),
    )

    # Dummy pass with data to infer input shapes (torchlayers dependency)
    model(train.shape)  # Unsqueeze to add batch
    print(f"Model to be trained:\n{model}\n")

    # Model is needed to register it's params for transport & proximity loss functions
    loss = nn.loss.get(args, model)
    optimizer = nn.optimizer.get(hyperparams, model)

    print(
        "Per epoch model accuracy and loss will be logged using Tensorboard.\n"
        f"Log directory: {args.tensorboard}\n"
    )
    writer = SummaryWriter(log_dir=args.tensorboard)
    train_gatherer = nn.metrics.get(
        writer, train, stage="Train", tasks=len(train_datasets), input_type=args.input,
    )
    validation_gatherer = nn.metrics.get(
        writer, validation, stage="Validation", tasks=len(validation_datasets), input_type=args.input
    )

    # Save best model
    checkpointer = None

    train_pass = nn.passes.Train(model, loss, optimizer, args.cuda)
    validation_pass = nn.passes.Validation(model, loss, args.cuda)
    train_loop = nn.train.get_loop(train_pass, train, hyperparams)
    validation_loop = nn.train.get_loop(validation_pass, validation, hyperparams)

    # Run training and validation
    for epoch in range(hyperparams["epochs"]):
        print(
            f"=============================== TRAINING {epoch} ==================================="
        )
        _dev_utils.run(train_loop, train_gatherer, epoch, True, checkpointer)
        print(
            f"============================== VALIDATION {epoch} =================================="
        )
        _dev_utils.run(
            validation_loop, validation_gatherer, epoch, False, checkpointer
        )
    print(f"Final model, saving at: {args.save}")
    torch.save(model, args.save)
