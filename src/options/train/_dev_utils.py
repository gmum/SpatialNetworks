import torch

import nn


def headline(epoch: int, train: bool):
    """Print headline of either training or validation."""
    headline = "TRAINING" if train else "VALIDATION"
    print(f"{headline} | Epoch: {epoch}")


def run(loop, gatherer, epoch, train: bool, checkpointer=None):
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
    if not train and checkpointer is not None:
        checkpointer(results["accuracy"])
    print(
        "===================================END========================================"
    )
