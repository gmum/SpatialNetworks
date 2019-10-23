"""
This module contains metrics used for training and evaluation of neural networks.
Is used for `train` and `score` subcommands of main.

If you want to perform additional measures of performance, please place them here.

"""

import abc


def get(writer, dataset, stage, tasks):
    """Based on user input return metrics attached to the network.

    Following metrics will be returned:
    - Accuracy (overall, no matter the task)
    - Loss
    - Per task accuracies (useful for `score` subparser command)

    All of them will log their data within Tensorboard as well.


    Parameters
    ----------
    writer : torch.utils.temsorboard.SummaryWriter
            Writer responsible for logging values.
    dataset: torch.utils.data.Dataset
            Dataset with `len` method (used for calculation of number of samples).
    stage: str
            Under which name results will be logged by Tensorboard. Usually
            it's something like train, validation, test. Case insensitive,
            will be automatically capitalized.
    tasks: int
            How many tasks were specified by the user.

    Returns
    -------
    Gather
            Gathering metrics containing all necessary information.

    """

    return Gather(
        # Each metric logs into Tensorboard under name
        accuracy=Accuracy(
            writer, name=f"Accuracy/{stage.capitalize()}", samples=len(dataset) * tasks
        ),
        loss=Loss(
            writer, name=f"Loss/{stage.capitalize()}", samples=len(dataset) * tasks
        ),
        # Accuracy per each task
        **{
            f"task{index}_accuracy": PerTaskAccuracy(
                writer,
                name=f"AccuracyTask{index}/{stage.capitalize()}",
                samples=len(dataset),
                task=index,
            )
            for index in range(tasks)
        },
    )


# Calculate mean and write it to tensorboard
class TensorboardMean:
    """Take mean of some value and log this value under specified name.

    Can be considered as convenience base class.
    Mean accuracy, mean loss and other could easily base off this class.

    Inheriting classes should overload `__call__` magic method.

    Parameters
    ----------
    writer : torch.utils.temsorboard.SummaryWriter
            Writer responsible for logging values.
    name: str
            Named under which values will be logged into Tensorboard.
    samples: int
            Total count of samples (used in division when obtaining results)

    """

    def __init__(self, writer, name: str, samples: int):
        self.writer = writer
        self.name: str = name

        self.score = 0
        self.step: int = 0
        # Samples iteratively? Minor
        self.samples: int = samples

    @abc.abstractmethod
    def __call__(self, output):
        """Register values from learning pass within Summary and `self.score`.

        Output is whatever is yielded from data passes (see `nn.pass` module).
        """
        pass

    def get(self):
        """Retrieve mean of results after keeping them within metric."""
        result = self.score / self.samples
        self.writer.add_scalar(self.name, result, self.step)
        self.step += 1
        self.score = 0
        return result


class Loss(TensorboardMean):
    """Calculate mean loss of neural network."""

    def __call__(self, output):
        loss, _, _, = output

        self.score += loss.item()


class Accuracy(TensorboardMean):
    """Calculate mean accuracy of neural network predictions across all tasks.

    For MultiOutput (e.g. mixup or concatenation) output of the final layer
    has to be reshaped from `(batch, task * classes)` into `(batch, task, labels)`
    and it's done automatically in this function.

    """

    def __call__(self, output):
        _, y_pred, y_true, = output
        if len(y_true.shape) > 1:
            y_pred = y_pred.reshape(y_true.shape[0], -1, y_true.shape[1])

        self.score += (y_pred.argmax(dim=1) == y_true).float().sum()


class PerTaskAccuracy(TensorboardMean):
    """Calculate mean accuracy of only specific task.

    For MultiOutput (e.g. mixup or concatenation) output of the final layer
    has to be reshaped from `(batch, task * classes)` into `(batch, task, labels)`
    and it's done automatically in this function.

    Parameters
    ----------
    writer : torch.utils.temsorboard.SummaryWriter
            Writer responsible for logging values.
    name: str
            Named under which values will be logged into Tensorboard.
    samples: int
            Total count of samples (used in division when obtaining results)
    task: int
            Index of task with which output tensor will be sliced.

    """

    def __init__(self, writer, name: str, samples: int, task: int):
        super().__init__(writer, name, samples)
        self.task: int = task

    def __call__(self, output):
        _, y_pred, y_true, = output
        if len(y_true.shape) > 1:
            y_pred = y_pred.reshape(y_true.shape[0], -1, y_true.shape[1])
            y_pred = y_pred[..., self.task]

        y_true = y_true[..., self.task]
        self.score += (y_pred.argmax(dim=1) == y_true).float().sum()


class Gather:
    """Gather all metrics and run/get them all with single call.

    `get` returns dictionary with name of each metrics for easier parsing.

    Parameters
    ----------
    *metrics: typing.Callable
            One of the above metrics.

    """

    def __init__(self, *_, **metrics):
        self.metrics = metrics

    def __call__(self, output):
        for metric in self.metrics.values():
            metric(output)

    def get(self):
        return {name: metric.get() for name, metric in self.metrics.items()}


def print_results(output):
    "Conveniently print results to stdout."
    for key, value in output.items():
        print(f"{key}: {value}")
