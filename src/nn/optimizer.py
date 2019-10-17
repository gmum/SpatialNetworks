import torch


def get(hyperparams, model):
    return getattr(torch.optim, hyperparams["optimizer_name"].capitalize())(
        model.parameters(), **hyperparams["optimizer_args"]
    )
