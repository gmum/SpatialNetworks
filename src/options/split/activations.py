import copy
import pathlib

import torch
import torchfunc

###############################################################################
#
#                               TO BE FIXED
#
###############################################################################


def get_data(data):
    for task in pathlib.Path(data).glob("*"):
        yield [torch.load(layer).cpu() for layer in sorted(list(task.glob("*")))]


def get_max_activations(data):
    return list(
        map(
            lambda tensor: torch.argmax(tensor, dim=0),
            (torch.stack(layers) for layers in zip(*data)),
        )
    )[
        1:
    ]  # Ignore first recorded activation as it's input related


def slice_model(model, max_activations):
    models = []
    # Iterate over tasks
    for i in range(len(max_activations)):
        new_model = copy.deepcopy(model)
        for max_activation, layer in zip(
            max_activations,
            (layer for layer in new_model.modules() if hasattr(layer, "weight")),
        ):
            layer.weight *= (max_activation == i).unsqueeze(dim=1).to(layer.weight)

        models.append(new_model)
    return models


def save_models(args, models):
    save_path = pathlib.Path(args.save)
    save_path.mkdir(parents=True, exist_ok=True)
    for i, model in enumerate(models):
        model_path = save_path / f"task{i}.pt"
        print(f"Saving task {i} model at {model_path}")
        torch.save(model, model_path)


def run(args):
    print("Getting data...")
    data = list(get_data(args.data))
    print("Processing activations...")
    max_activations = get_max_activations(data)

    print("Loading model...")
    model = torch.load(args.model)
    model.eval()
    torchfunc.module.freeze(model)

    print("Slicing model...")
    models = slice_model(model, max_activations)
    save_models(args, models)
