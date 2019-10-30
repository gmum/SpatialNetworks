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
        hidden_masks = list(reversed(reversed_masks[:-1]))  # drop the last mask
        masks = hidden_masks + [output_mask]
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
        mask_idx = 0
        model = get_model(args)
        for module in model.modules():
            if nn.layers.spatial(module):
                if mask_idx == 0:  # skip the first layer
                    mask_idx += 1
                    continue

                masker.apply(module.weight.data, masks[mask_idx], task)
                mask_idx += 1
        torch.save(model, path / f"{task}.pt")
