import torch


def _mean(previous, current):
    return torch.abs(previous) + torch.abs(current)


def _variance(previous, current):
    if len(previous.size()) != 2:
        return torch.stack((previous + current, previous + current * current), dim=0)
    previous[0] += current
    previous[1] += current * current
    return previous


def get(args):
    if args.reduction.lower() == "mean":
        return _mean
    return _variance
