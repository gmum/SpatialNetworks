# Calculate mean and write it to tensorboard
class TensorboardMean:
    def __init__(self, writer, name: str, samples: int):
        self.writer = writer
        self.name: str = name

        self.score = 0
        self.step: int = 0
        self.samples: int = samples

    def get(self):
        result = self.score / self.samples
        self.writer.add_scalar(self.name, result, self.step)
        self.step += 1
        self.score = 0
        return result


class Loss(TensorboardMean):
    def __call__(self, output):
        loss, _, _, = output

        self.score += loss.item()


class Accuracy(TensorboardMean):
    def __call__(self, output):
        _, y_pred, y_true, = output
        if len(y_true.shape) > 1:
            y_pred = y_pred.reshape(y_true.shape[0], -1, y_true.shape[1])

        self.score += (y_pred.argmax(dim=1) == y_true).float().mean()


class PerTaskAccuracy(TensorboardMean):
    def __init__(self, writer, name: str, samples: int, task: int):
        super().__init__(writer, name, samples)
        self.task: int = task

    def __call__(self, output):
        _, y_pred, y_true, = output
        if len(y_true.shape) > 1:
            y_pred = y_pred.reshape(y_true.shape[0], -1, y_true.shape[1])
            y_pred = y_pred[..., self.task]

        y_true = y_true[..., self.task]
        self.score += (y_pred.argmax(dim=1) == y_true).float().mean()


class Gather:
    def __init__(self, **metrics):
        self.metrics = metrics

    def __call__(self, output):
        for metric in self.metrics.values():
            metric(output)

    def get(self):
        return {name: metric.get() for name, metric in self.metrics.items()}


def get(writer, dataset, hyperparams, stage, tasks):
    samples = len(dataset) // hyperparams["batch"]
    return Gather(
        # Each metric logs into Tensorboard under name
        accuracy=Accuracy(
            writer,
            name=f"Accuracy/{stage}",
            samples=len(dataset) // hyperparams["batch"],
        ),
        loss=Loss(
            writer, name=f"Loss/{stage}", samples=len(dataset) // hyperparams["batch"]
        ),
        # Accuracy per each task
        **{
            f"task{index}_accuracy": PerTaskAccuracy(
                writer, name=f"AccuracyTask{index}", samples=samples, task=index
            )
            for index in range(tasks)
        },
    )


def print_results(output):
    for key, value in output.items():
        print(f"{key}: {value}")
