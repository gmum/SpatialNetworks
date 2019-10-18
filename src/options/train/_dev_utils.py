import torch

import nn


def headline(epoch: int, train: bool):
    """Print headline of either training or validation."""
    headline = "TRAINING" if train else "VALIDATION"
    print(f"{headline} | Epoch: {epoch}")


def run(loop, gatherer, epoch, checkpointer, train: bool):
    """Run either training or validation loop and print gathered results.

    Additionally save model if it's validation is better than current best.
    """
    for result in loop():
        gatherer(result)
    results = gatherer.get()
    print(
        "=================================RESULTS======================================"
    )
    nn.metrics.print_results(results)
    if not train:
        checkpointer(results["accuracy"])
    print(
        "===================================END========================================"
    )
