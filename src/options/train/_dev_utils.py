import torch

import nn


def headline(epoch: int, train: bool):
    headline = "TRAINING" if train else "VALIDATION"
    print(f"{headline} | Epoch: {epoch}")


def run(loop, gatherer, epoch, save_model, train: bool):
    for result in loop():
        gatherer(result)
    results = gatherer.get()
    print(
        "=================================RESULTS======================================"
    )
    nn.metrics.print_results(results)
    if not train:
        save_model(results["accuracy"])
    print(
        "===================================END========================================"
    )
