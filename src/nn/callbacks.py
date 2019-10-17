import dataclasses
import pathlib
import typing

import torch

from ._utils import print_verbose


@dataclasses.dataclass
class SaveBest:
    """Save best model according to specified metric.

    Parameters
    ----------
    path: pathlib.Path
            Path where model will be saved (has to be file, not folder).
    model: torch.nn.Module
            Module to save
    operator : typing.Callable[float, float]
            Function comparing two values - current metric and best metric so far.
            If true, save new model. One can user Python's standard operator library
            for this argument.
    verbose: bool
            If true, print to stdout informations about saving the model.

    """

    path: pathlib.Path
    model: torch.nn.Module
    operator: typing.Callable
    verbose: bool = True

    def __post_init__(self,):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._best = None

    def __call__(self, metric):
        """On epoch end, evaluate network's performance.

        This object is attached to trainer.

        Parameters
        ----------
        trainer : ignite.engine.Engine
                Trainer engine, to which this callback is attached.
        """
        if self._best is None or self.operator(metric, self._best):
            self._best = metric
            print_verbose(f"New best model, saving at: {self.path}", self.verbose)
            torch.save(self.model, self.path)
