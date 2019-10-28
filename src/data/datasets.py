import typing

import torch
import torchdata
import torchvision

from . import samplers


def get(args, *datasets):
    """Get appropriate dataset from user provided arguments.

    Merge multiple datasets into one (with sampling) for easier usage down the
    road.

    Parameters
    ----------
    args: argparse.Namespace
            argparse.ArgumentParser().parse() return value. User provided arguments.
    *datasets: torch.utils.data.Dataset instances
            Datasets to be merged into one

    Returns
    -------
    torch.utils.data.Dataset
            Dataset merging all datasets according to `input` argument (for example
            input concatenation).

    """
    # Validate whether they are one of available choices.
    if args.input.lower() == "sequential":
        if args.task:
            return Sequential(args.labels, samplers.TaskSampler, *datasets)
        return Sequential(args.labels, samplers.DatasetRandomSampler, *datasets)
    elif args.input.lower() == "mix":
        return Mix(args.labels, samplers.JoinedSampler, *datasets)
    return Stacked(args.labels, samplers.JoinedSampler, *datasets)


# Could be standard dataset as well, but left it as generator for more concise API
# Change if you wish
class Sequential(torch.utils.data.IterableDataset):
    """Provide samples sequentially.

    This dataset is an instance of IterableDataset for unified API-like structure
    with Mix and Stacked (which have to be IterableDataset).

    Parameters
    ----------
    labels: int
            How many labels will this dataset use. Every label will be divided by modulo.
            This allows datasets with more classes to be considered as multiple datasets
            with up to `labels` distinct classification labels.
            For example: Dataset has 50 labels, we want 10, so effectively those
            will be transformed into 5 * [0,9] labels via modulo operation.
    sampler_class : torch.utils.data.Sampler
            Instance of Sampler class (probably from `samplers` module).
            Specifies how we will iterate over data.
    *datasets : torch.utils.data.Dataset
            Varargs containing datasets to be merged

    Yields
    ------
    Tensor, int
            Image, label pair from one of the datasets (exact scheme described by sampler).

    """

    def __init__(self, labels: int, sampler_class, *datasets):
        self.labels: int = labels
        self.sampler_class = sampler_class
        self.datasets = datasets

        # lengths will be useful to know which dataset we are going to index.
        self.lengths = [0] + torch.cumsum(
            torch.tensor([len(dataset) for dataset in self.datasets]), dim=0
        ).tolist()
        self._last_label = None

    def __iter__(self):
        # Sampler has to be resetted after each full pass through data
        # Can be considered data shuffle
        self.reset()
        for index in self._sampler:
            # Which dataset should be queried now based on sampler
            # E.g. 0 for MNIST, 1 for Fashion-MNIST etc.
            dataset_index = next(i for i, v in enumerate(self.lengths) if v > index) - 1
            image, label = self.datasets[dataset_index][
                index - self.lengths[dataset_index]
            ]
            # If there are more labels in dataset than specified (say 50)
            # Take modulo and assume those are the same.
            # E.g. Dataset with 50 labels, when 10 specified, would effectively
            # become 5 separate datasets with 10 labels
            label %= self.labels
            self._last_label = label
            self._last_dataset = dataset_index
            yield image, label

    def __len__(self):
        return sum(len(dataset) for dataset in self.datasets)

    def reset(self):
        self._sampler = self.sampler_class(*self.datasets)

    # Used in shape inference, shape of single sample (without labels)
    @property
    def shape(self):
        """Shape of single image.

        Convenience method, useful for shape inference in modules.
        Only image is returned as that's all what's needed by torchlayers.

        Returns
        -------
        torch.Tensor
                Image (with `1` batch size)
        """
        image, _ = self.datasets[0][0]
        return image.unsqueeze(dim=0)

    # Below properties are used during recording of activations
    @property
    def label(self):
        return self._last_label

    @property
    def dataset(self):
        return self._last_dataset

    @property
    def inner_datasets(self):
        return len(self.datasets)


class _JoinedDataset(torch.utils.data.IterableDataset):
    """Provide joined samples (by some, later to be specified, method).

    This dataset is an instance of IterableDataset as `torch.utils.data.Dataset`
    __getitem__ magic method cannot receive tensor as index (only `int` or `str`).

    Parameters
    ----------
    labels: int
            How many labels will this dataset use. Every label will be divided by modulo.
            This allows datasets with more classes to be considered as multiple datasets
            with up to `labels` distinct classification labels.
            For example: Dataset has 50 labels, we want 10, so effectively those
            will be transformed into 5 * [0,9] labels via modulo operation.
    sampler_class : torch.utils.data.Sampler
            Instance of Sampler class (probably from `samplers` module).
            Specifies how we will iterate over data.
            Special sampler: `samplers.JoinedSampler` should be used.
    *datasets : torch.utils.data.Dataset
            Varargs containing datasets to be merged

    Yields
    ------
    Tensor, int
            Image, label pair from one of the datasets (exact scheme described by sampler).

    """

    def __init__(
        self, operation: typing.Callable, labels: int, sampler_class, *datasets
    ):
        self.labels: int = labels
        self.sampler_class = sampler_class
        self.datasets = datasets

        self._length = max(map(len, datasets))
        self._operation = operation

    def __iter__(self):
        # Sampler has to be reinstantiated after each full pass through data in order to shuffle
        self.reset()
        for indices in self._sampler:
            X, y = [], []
            for dataset, index in zip(self.datasets, indices):
                sample = dataset[index]
                X.append(sample[0])
                y.append(sample[1])
            y = torch.tensor(y)
            y %= self.labels
            yield self._operation(X), y

    def __len__(self):
        return self._length

    def reset(self):
        self._sampler = self.sampler_class(*self.datasets)


# Adding parameters like in real mix for each dataset?
# E.g. 1.5 MNIST and 0.2 Fashion-MNIST, could make the task harder
class Mix(_JoinedDataset):
    """Joined dataset summing images."""

    def __init__(self, labels: int, sampler, *datasets):
        super().__init__(
            lambda X: torch.stack(X, dim=0).sum(dim=0), labels, sampler, *datasets
        )

    @property
    def shape(self):
        """Shape of single image.

        Convenience method, useful for shape inference in modules.
        Only image is returned as that's all what's needed by torchlayers.

        Returns
        -------
        torch.Tensor
                Image (with `1` batch size)
        """
        image, _ = self.datasets[0][0]
        return image.unsqueeze(dim=0)


class Stacked(_JoinedDataset):
    """Joined dataset stacking images."""

    def __init__(self, labels: int, sampler, *datasets):
        super().__init__(lambda X: torch.cat(X, dim=0), labels, sampler, *datasets)

    @property
    def shape(self):
        """Shape of single image.

        Convenience method, useful for shape inference in modules.
        Only image is returned as that's all what's needed by torchlayers.

        Returns
        -------
        torch.Tensor
                Images (with `1` batch size) concatenated along channels dimension.
        """
        images = []
        for _ in range(len(self.datasets)):
            image, _ = self.datasets[0][0]
            images.append(image)
        return torch.cat(images).unsqueeze(dim=0)
