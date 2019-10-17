import torchvision


def get_datasets(args, train):
    return [
        getattr(torchvision.datasets, name)(
            args.root,
            train=train,
            download=True,
            transform=torchvision.transforms.ToTensor(),
        )
        for name in args.datasets
    ]
