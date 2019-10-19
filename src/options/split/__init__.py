import importlib
import sys

from . import _dev_utils, activations
from .backward import greedy, probability, rescale


def run(args):
    module = getattr(sys.modules[__name__], args.method.lower())

    model = _dev_utils.get_model(args)
    tasks = _dev_utils.get_tasks(args, model)

    masker = getattr(module, "Masker")(args.labels)
    _dev_utils.generate_networks(args, tasks, masker)
