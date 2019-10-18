import torchvision


def get_datasets(args, train: bool):
    """Return user specified `torchvision` datasets.

    Please note those need to have the same signature;
    If you were to change that, you should manipulate this function accordingly
    (or make it into new module).

    Parameters
    ----------
    args: argparse.Namespace
            argparse.ArgumentParser().parse() return value. User provided arguments.
    train : bool
            Whether to return training or validation part of datasets.

    Returns
    -------
    List[torch.utils.data.Dataset]

    """
    return [
        getattr(torchvision.datasets, name)(
            args.root,
            train=train,
            download=True,
            transform=torchvision.transforms.ToTensor(),
        )
        for name in args.datasets
    ]
