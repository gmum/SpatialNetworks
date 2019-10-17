from . import _dev_utils, activations


def run(args):
    if args.data is not None:
        data = _dev_utils.divide_by_layer(list(_dev_utils.get_data(args.data)))
        activations.plot(_dev_utils.get_task_specific_activations(data), args)
    # Add standard 3D plotting etc.
    # else:
    #     print(
    #         "Currently only measured activation strength can be plotted.\n"
    #         "Plotting of spatial neural network will be integrated soon."
    #     )
