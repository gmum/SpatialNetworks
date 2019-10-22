import pathlib
import typing

import torch
import torchfunc

import nn


def get_model(args):
    """Load trained model to be splitted.

    Parameters
    ----------
    args: argparse.Namespace
            argparse.ArgumentParser().parse() return value. User provided arguments.

    Returns
    -------
    torch.nn.Module
        Frozen module in evaluation mode.

    """
    model = torch.load(args.model)
    model.eval()
    return torchfunc.module.freeze(model)


def get_tasks(args, model) -> int:
    """Helper calculating how many tasks were used during network training.

    Parameters
    ----------
    args: argparse.Namespace
            argparse.ArgumentParser().parse() return value. User provided arguments.
    model: torch.nn.Module
            Frozen module in evaluation mode.

    Returns
    -------
    int
            Number of tasks

    """
    return list(model.modules())[-1].weight.shape[0] // args.labels


def generate_networks(args, tasks, masker: typing.Callable):
    """Generate splitted per-task neural networks.

    Parameters
    ----------
    args: argparse.Namespace
            argparse.ArgumentParser().parse() return value. User provided arguments.
    tasks: int
            How many tasks were done within original network.
    masker:
            Object creating and applying masks to neural network layers

    Yields
    -------
    torch.nn.Module
            Module with per-task masks applied.

    """
    path = pathlib.Path(args.save)
    path.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        for task in range(tasks):
            model = get_model(args)
            masker.reset()
            for module in reversed(list(model.modules())):
                if hasattr(module, "weight") and nn.layers.spatial(module):
                    mask = masker(module)
                    masker.apply(module.weight.data, mask, task)
            torch.save(model, path / f"{task}.pt")
