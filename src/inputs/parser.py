import argparse

from ._dev_utils.parser import process_arguments
from .subparsers import plot, record, score, split, train


def get():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--cuda",
        required=False,
        action="store_true",
        help="Whether to use GPU instance.",
    )

    parser.add_argument(
        "--labels",
        type=int,
        required=True,
        help="How many labels are used for classification. If more classes in dataset available, "
        + "modulo will be taken from this dataset (e.g. 50 labels will become 5 datasets, 10 labels each).",
    )

    subparsers = parser.add_subparsers(help="Actions to perform:", dest="command")
    train(subparsers)
    record(subparsers)
    plot(subparsers)
    split(subparsers)
    score(subparsers)
    return process_arguments(parser.parse_args())
