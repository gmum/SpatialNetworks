import importlib
import sys

from . import _dev_utils, activations, greedy, probability, rescale


def run(args):
    module = getattr(sys.modules[__name__], args.method.lower())

    model = _dev_utils.get_model(args)
    tasks = _dev_utils.get_tasks(args, model)

    mask_creator = getattr(module, "MaskCreator")(args.labels)
    _dev_utils.generate_networks(args, tasks, mask_creator, getattr(module, "apply_mask"))
