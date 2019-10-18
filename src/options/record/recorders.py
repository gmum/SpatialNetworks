import functools
import pathlib
import typing

import torchfunc

import nn
import torchlayers

"""
Types whose inputs should be recorded using `torchfunc.hooks.recorders.ForwardPre`.

"""
RECORDER_TYPES = (
    torchlayers.Linear,
    torchlayers.Conv,
    nn.layers._SpatialConv,
    nn.layers._SpatialLinear,
)


def get(dataset, reduction) -> typing.List:
    """Get per-task layer activations.

    Those will record output from each layer as data samples pass
    and apply either running mean or running variance reduction.

    Parameters
    ----------
    dataset : torch.utils.data.Dataset
            Specific dataset containing information about last task returned.
            Has to be sequential, so most probably data.datasets.Sequential
            should be used here.
    reduction : typing.Callable
            Either running mean-like or running variance-like operation

    Returns
    -------
    typing.List[torchfunc.hooks.recorders.ForwardPre]
            Recorders recording inner state of neural network as data passes.
            One recorder is responsible for recording network activations
            of one task.

    """

    def condition(index):
        """If `True`, record given sample.

        Check whether last sample was from specified task.

        Parameters
        ----------
        index : int
                Task index to check against `dataset.dataset` (e.g. last used dataset/sample).

        Returns
        -------
        bool

        """
        return dataset.dataset == index

    return [
        torchfunc.hooks.recorders.ForwardPre(
            condition=functools.partial(condition, i), reduction=reduction
        )
        for i in range(dataset.inner_datasets)
    ]


def register(model, recorders) -> None:
    """Register recorders to the model.

    Recorders will gather per-task specific activations when data goes into
    neural network.

    Parameters
    ----------
    model: torch.nn.Module
            Module for which it's inner state (input activations of layers)
            will be recorded.
    recorders: typing.List[torchfunc.hooks.recorders.ForwardPre]
            Recorders to be registered

    """
    for task_recorder in recorders:
        task_recorder.modules(model, types=RECORDER_TYPES)


def save(recorders, args) -> None:
    """After recording and applying final operations, save per-task activations.

    Appropriate folder containing per-task activations data of each layer
    will be created if needed.

    Folder structure looks like (assuming data will be saved inside `recorded`):
    recorded/
        task1/
            1.pt
            2.pt
            ...
            N.pt (where N is the number of layer activations)
        task2/
            1.pt
            2.pt
            ...
            N.pt (where N is the number of layer activations)
        ...
        taskM/ (total number of tasks (separate datasets passed by user))
            1.pt
            2.pt
            ...
            N.pt (where N is the number of layer activations)


    Parameters
    ----------
    recorders: typing.List[torchfunc.hooks.recorders.ForwardPre]
            Recorders to be registered
    args: argparse.Namespace
            argparse.ArgumentParser().parse() return value. User provided arguments.

    """
    path = pathlib.Path(args.save)
    for index, task_recorder in enumerate(recorders):
        task_recorder.save(path / str(index), mkdir=True, parents=True, exist_ok=True)
