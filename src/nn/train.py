import dataclasses
import typing

import torch
import tqdm


# Yield whatever is returned from single pass through sample
# Essentially is an epoch (maybe should be called an epoch?)
@dataclasses.dataclass
class Loop:
    through: typing.Callable
    dataloader: torch.utils.data.DataLoader
    length: int

    def __call__(self):
        for sample in tqdm.tqdm(self.dataloader, total=self.length):
            yield self.through(sample)


def get_loop(single_pass, dataset, hyperparams):
    return Loop(
        single_pass,
        torch.utils.data.DataLoader(
            dataset,
            shuffle=False,
            batch_size=hyperparams["batch"],
            pin_memory=single_pass.cuda,
        ),
        len(dataset) // hyperparams["batch"],
    )
