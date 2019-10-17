import typing

import torch

import torchscripts

from .utils.record import (apply_final_operations, gather_data, get_model,
                           get_recorders, register_recorders)


def run(args, dataset, custom_layer_types: typing.List):
    model = get_model(args)
    recorders = get_recorders(dataset, args)
    register_recorders(model, recorders, custom_layer_types)
    gather_data(model, dataset, args)
    apply_final_operations(recorders, args)

    return tuple(recorders)
