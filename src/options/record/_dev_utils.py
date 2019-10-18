import torch
import torchfunc
import tqdm


def get_model(args):
    """Load and freeze model.

    Parameters
    ----------
    args: argparse.Namespace
            argparse.ArgumentParser().parse() return value. User provided arguments.

    Returns
    -------
    torch.nn.Module

    """
    model = torch.load(args.model)
    model.eval()
    torchfunc.module.freeze(model)
    if args.cuda:
        return model.cuda()
    return model


def record_state(model, dataset, args):
    """Record neural network state when data passes through it.

    Output from all layers will be recorded after this operation.

    Parameters
    ----------
    model: torch.nn.Module
            Model whose parameters will be, possibly, used by `transport` and
            `proximity` loss
    dataset: torch.utils.data.Dataset
            Dataset with `len` method (used for calculation of number of samples).
    args: argparse.Namespace
            argparse.ArgumentParser().parse() return value. User provided arguments.

    Returns
    -------
    torch.nn.Module

    """
    for element, _ in tqdm.tqdm(dataset, total=len(dataset)):
        if args.cuda is not None:
            element = element.cuda()

        model(element)
