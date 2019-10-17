import pathlib

import torch


def get_data(data):
    for task in pathlib.Path(data).glob("*"):
        yield [torch.load(layer).cpu() for layer in sorted(list(task.glob("*")))]


def divide_by_layer(data):
    by_task = [[] for _ in range(len(data[0]))]
    for task in data:
        for index, layer in enumerate(task):
            by_task[index].append(layer)
    return list(map(lambda tensor: torch.stack(tensor, axis=0), by_task))


def get_task_specific_activations(data):
    tasks = [[] for _ in range(data[0].shape[0])]
    for layer in data:
        for index, task in enumerate(layer):
            current_task = layer[index]
            rest_slice = torch.full((layer.shape[0],), True).bool()
            rest_slice[index] = False
            # Maximum activation of other tasks
            rest, _ = torch.max(layer[rest_slice], dim=0)
            tasks[index].append(torch.clamp(current_task - rest, min=0))
    return tasks
