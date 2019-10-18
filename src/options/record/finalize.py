def _variance(mean_variance, samples):
    """Perform final variance calculations.

    `mean_variance` and `samples` were pre-recorded with
    `options.record.reduction._variance`.

    This operation only calculates total running variance.

    This function is applied on per-recording basis (per layer to be exact).

    Parameters
    ----------
    mean_variance : Tuple[torch.Tensor]
            Running tensor containing running mean and running variance
    samples : int
            How many samples passed through this tensor (used for taking the mean).

    Returns
    -------
    torch.Tensor
            Tensor containing per-neuron variance (exact same shape as output layer)

    """
    mean = mean_variance[0] / samples
    variance = mean_variance[1]
    variance /= samples
    variance -= mean * mean
    return variance


def _get(args):
    """Get appropriate operation (either mean or variance) based on user input.

    Parameters
    ----------
    args: argparse.Namespace
            argparse.ArgumentParser().parse() return value. User provided arguments.

    Returns
    -------
    typing.Callable[[torch.Tensor, int], torch.Tensor]
            Callable getting per-layer recorded activations and number of samples
            which passed through this layer. Returns per layer running mean or variance.

    """
    if args.reduction.lower() == "mean":
        return lambda data, samples: data / samples
    return _variance


def apply(recorders, args) -> None:
    """Apply either running mean or running variance.

    This function is applied after `torchfunc.hooks.recorders` recorded and reduced
    activations appropriately.

    Modifies recorded data in-place.

    Parameters
    ----------
    recorders: typing.List[torchfunc.hooks.recorders.ForwardPre]
            Recorders containing ForwardPre data.
    args: argparse.Namespace
            argparse.ArgumentParser().parse() return value. User provided arguments.

    """
    operation = _get(args)
    for recorder in recorders:
        recorder.apply_sample(operation)
