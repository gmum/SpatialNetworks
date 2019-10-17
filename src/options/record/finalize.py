def _variance(mean_variance, samples):
    mean = mean_variance[0] / samples
    variance = mean_variance[1]
    variance /= samples
    variance -= mean * mean
    return variance


def _get(args):
    if args.reduction.lower() == "mean":
        return lambda data, samples: data / samples
    return _variance


def apply(recorders, args):
    operation = _get(args)
    for recorder in recorders:
        recorder.apply_sample(operation)
