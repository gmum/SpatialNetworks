import torch


def _mean(previous, current):
    """Sum two elements with absolute value.

    This operation will be applied as data passes through layer.
    Within `finalize` you can find the final mean calculation, this is
    kind of "running">

    Parameters
    ----------
    previous : torch.Tensor
            Currently hold sum of activations (layer statistics)
    current : type
            New incoming activations for new data sample.

    Returns
    -------
    torch.Tensor

    """
    return torch.abs(previous) + torch.abs(current)


def _variance(previous, current):
    """Calculate mean and variance (as both are needed in final operation).

    Parameters
    ----------
    previous : torch.Tensor
            Currently hold sum of activations (layer statistics)
    current : type
            New incoming activations for new data sample.

    Returns
    -------
    torch.Tensor

    """
    if len(previous.size()) != 2:
        return torch.stack((previous + current, previous + current * current), dim=0)
    previous[0] += current
    previous[1] += current * current
    return previous


def get(args):
    """Return either running mean or variance operation.

    Parameters
    ----------
    args: argparse.Namespace
            argparse.ArgumentParser().parse() return value. User provided arguments.

    Returns
    -------
    typing.Callable


    """
    if args.reduction.lower() == "mean":
        return _mean
    return _variance
