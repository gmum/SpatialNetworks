import torch
from torch.utils.data import RandomSampler, Sampler, SubsetRandomSampler


# Iterate over every dataset randomly
class DatasetRandomSampler(SubsetRandomSampler):
    """Standard Random Sampler but return indices from all datasets.

    Example:
    --------

    For datasets of length [10000, 10000, 20000] will return sampler
    yielding indices [0, 40000] (e.g. summed).

    Yields
    ------
    int
            Index into one of datasets.

    """

    def __init__(self, *datasets):
        super().__init__(torch.arange(sum(map(len, datasets))))


# Iterate randomly over first dataset, after that randomly through the second and so on...
class TaskSampler(Sampler):
    """Iterate randomly but in order of tasks provided.

    Example:
    --------

    For datasets of length [10000, 10000, 20000] will return sampler
    yielding random indices from range [0, 10000], followed by [100000, 200000]
    and lastly [20000, 40000].

    Yields
    ------
    int
            Index into one of datasets.

    """

    def __init__(self, *datasets):
        self.samplers = [RandomSampler(dataset) for dataset in datasets]
        self._length = sum(len(sampler) for sampler in self.samplers)
        self._cumulative_lengths = [0] + torch.cumsum(
            torch.tensor([len(sampler) for sampler in self.samplers]), dim=0
        ).tolist()[:-1]

    def __iter__(self):
        for sampler, shift in zip(self.samplers, self._cumulative_lengths):
            for index in sampler:
                yield index + shift

    def __len__(self):
        return self._length


# Return multiple indices, each for one dataset
class JoinedSampler(Sampler):
    """Return multiple indices each for one dataset.

    Used by concatenation and mixup inputs (or any other where elements)
    from datasets are needed at the same time.

    Datasets will be sampled with replacement if they don't have the same
    number of elements within them.

    As this sampler yields tuples, it cannot be used with Dataset's __getitem__,
    hence it is connected with dataset (see datasets module and _JoinedDataset base class).

    Example:
    --------

    Yields
    ------
    Tuple[int]
            Tuple containing indices into each of the dataset.

    """

    # Each dataset shorter than the longest will be upsampled so number of samples will be equal.
    # This may induce overfitting on some datasets and underfitting on others.
    @classmethod
    def _with_replacement_except_for_longest(cls, dataset, longest):
        if len(dataset) != longest:
            return {"replacement": True, "num_samples": longest}
        return {}

    def __init__(self, *datasets):
        self._length = max(map(len, datasets))
        self.samplers = [
            RandomSampler(
                dataset,
                **JoinedSampler._with_replacement_except_for_longest(
                    dataset, self._length
                )
            )
            for dataset in datasets
        ]

    def __iter__(self):
        yield from zip(*self.samplers)

    def __len__(self):
        return self._length
