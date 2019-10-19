import pathlib
import typing

import torch
import torchfunc


def get_model(args):
    model = torch.load(args.model)
    model.eval()
    return torchfunc.module.freeze(model)


def get_tasks(args, model) -> int:
    return list(model.modules())[-1].weight.shape[0] // args.labels


def generate_networks(args, tasks, masker: typing.Callable):
    path = pathlib.Path(args.save)
    path.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        for task in range(tasks):
            model = get_model(args)
            for module in reversed(list(model.modules())):
                if hasattr(module, "weight"):
                    mask = masker(module)
                    masker.apply(module.weight.data, mask, task)
            torch.save(model, path / f"{task}.pt")
