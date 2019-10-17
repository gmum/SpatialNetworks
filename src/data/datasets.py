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
            <return description>

    """
    # Validate whether they are one of available choices.
    if args.input.lower() == "sequential":
        if args.task:
            return Sequential(args.labels, samplers.TaskSampler, *datasets)
        return Sequential(args.labels, samplers.DatasetRandomSampler, *datasets)
    elif args.input.lower() == "mixup":
        return MixUp(args.labels, samplers.JoinedSampler, *datasets)
    return Stacked(args.labels, samplers.JoinedSampler, *datasets)


# Could be standard dataset as well, but left it as generator for more concise API
# Change if you wish
class Sequential(torch.utils.data.IterableDataset):
    """Sequential (short summary)."""

    def __init__(self, labels: int, sampler_class, *datasets):
        """<short summary>.

        <extended summary>

        Parameters
        ----------
        {{_indent}}labels:{{_indent}} : type
                <argument description>
        sampler_class : type
                <argument description>
        *datasets : type
                <argument description>

        Returns
        -------
        type
                <return description>

        """
        self.labels: int = labels
        self.sampler_class = sampler_class
        self.datasets = datasets

        self.lengths = [0] + torch.cumsum(
            torch.tensor([len(dataset) for dataset in self.datasets]), dim=0
        ).tolist()
        self._last_label = None

    def __iter__(self):
        """__iter__ (short summary)."""
        # Sampler has to be resetted after each full pass through data
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
        """__len__ (short summary)."""
        return sum(len(dataset) for dataset in self.datasets)

    def reset(self):
        """reset (short summary)."""
        self._sampler = self.sampler_class(*self.datasets)

    # Used in shape inference, shape of single sample (without labels)
    @property
    def shape(self):
        """shape (short summary)."""
        image, _ = self.datasets[0][0]
        return image

    # Below properties are used during recording of activations
    @property
    def label(self):
        """label (short summary)."""
        return self._last_label

    @property
    def dataset(self):
        """dataset (short summary)."""
        return self._last_dataset

    @property
    def inner_datasets(self):
        """inner_datasets (short summary)."""
        return len(self.datasets)


class _JoinedDataset(torch.utils.data.IterableDataset):
    """_JoinedDataset (short summary)."""

    def __init__(
        self, operation: typing.Callable, labels: int, sampler_class, *datasets
    ):
        """<short summary>.

        <extended summary>

        Parameters
        ----------
        {{_indent}}operation:{{_indent}} : type
                <argument description>
        {{_indent}}labels:{{_indent}} : type
                <argument description>
        sampler_class : type
                <argument description>
        *datasets : type
                <argument description>

        Returns
        -------
        type
                <return description>

        """
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
            yield self._operation(X), torch.tensor(y)

    def __len__(self):
        return self._length

    def reset(self):
        self._sampler = self.sampler_class(*self.datasets)


# Adding parameters like in real mixup for each dataset?
# E.g. 1.5 MNIST and 0.2 Fashion-MNIST, could make the task harder
class MixUp(_JoinedDataset):
    def __init__(self, labels: int, sampler, *datasets):
        super().__init__(
            lambda X: torch.stack(X, dim=0).sum(dim=0), labels, sampler, *datasets
        )

    @property
    def shape(self):
        image, _ = self.datasets[0][0]
        return image


class Stacked(_JoinedDataset):
    def __init__(self, labels: int, sampler, *datasets):
        super().__init__(lambda X: torch.cat(X, dim=0), labels, sampler, *datasets)

    @property
    def shape(self):
        images = []
        for _ in range(len(self.datasets)):
            image, _ = self.datasets[0][0]
            images.append(image)
        return torch.cat(images, dim=0)
