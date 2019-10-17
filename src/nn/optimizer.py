import torch


def get(hyperparams, model):
    """Based on user input return appropriate optimizer.

    Parameters
    ----------
    hyperparams: Dict
            Dictionary containing hyperparameters specified by user with
            --hyperparams flag.
    model: torch.nn.Module
            Model whose parameters will be, possibly, used by `transport` and
            `proximity` loss

    Returns
    -------
    torch.optim.Optimizer
            Concrete instance of built-in `torch.optim.Optimizer`

    """
    return getattr(torch.optim, hyperparams["optimizer_name"].capitalize())(
        model.parameters(), **hyperparams["optimizer_args"]
    )
