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

def get_masks(args, tasks, masker: typing.Callable):
    """Generate masks of task assignment for each layer

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
    list
            List of torch tensors representing the masks.

    """
    with torch.no_grad():
        model = get_model(args)
        reversed_modules = reversed(list(model.modules()))
        output_mask = torch.tensor(list(range(tasks)) * args.labels)
        reversed_masks = [
            masker(module) for module in reversed_modules
            if nn.layers.spatial(module)
        ]
        masks = list(reversed(reversed_masks)) + [output_mask]
    return masks


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

    masks = get_masks(args, tasks, masker)

    for task in range(tasks):
        model = get_model(args)
        layer_idx = 0
        for module in model.modules():
            if nn.layers.spatial(module):
                print(layer_idx, module, masks[layer_idx].shape)
                if layer_idx in args.where:
                    masker.apply(module.weight.data, masks[layer_idx], task)
                layer_idx += 1
        torch.save(model, path / f"{task}.pt")
