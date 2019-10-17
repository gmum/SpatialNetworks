import torch
from torch.utils.data import RandomSampler, Sampler, SubsetRandomSampler


# Iterate over every dataset randomly
class DatasetRandomSampler(SubsetRandomSampler):
    def __init__(self, *datasets):
        super().__init__(torch.arange(sum(map(len, datasets))))


# Iterate randomly over first dataset, after that randomly through the second and so on...
class TaskSampler(Sampler):
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
