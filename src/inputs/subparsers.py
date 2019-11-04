import argparse
import tempfile


def train(subparsers) -> None:
    subparser = subparsers.add_parser(
        "train",
        help="Fit neural network",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    subparser.add_argument(
        "--hyperparams",
        required=True,
        help="JSON file containing hyperparameters.\n"
        "Defaults probably shouldn't be changed as those are more related to hyperparameters search rather than experiments.\n",
    )

    subparser.add_argument(
        "--datasets",
        required=False,
        default=["MNIST", "FashionMNIST", "KMNIST", "QMNIST"],
        nargs="+",
        help="Name of torchvision datasets used in experiment.\nRestrictions and traits:\n"
        "- Provided datasets have to be proper object from torchvision.datasets.\n"
        "- Provided datasets need the same input shape. \n"
        "- Provided datasets CAN HAVE varying number of labels (modulo will be taken).\n"
        "This option is case sensitive.\n"
        "Default: ['MNIST', 'FashionMNIST', 'KMNIST', 'QMNIST']",
    )

    subparser.add_argument(
        "--root",
        required=False,
        default=tempfile.gettempdir(),
        help="Where downloaded datasets will be saved. By default inside your temporary folder.",
    )

    subparser.add_argument(
        "--layers",
        required=True,
        type=int,
        nargs="+",
        help="Size of each hidden layer specified as integer (specify as many as you want).\n"
        "Length of layers has to be greater than maximum index specified by '--where' option",
    )

    subparser.add_argument(
        "--where",
        required=False,
        default=None,
        type=int,
        nargs="+",
        help="Indices of layers to which we apply the spatial costs.\n"
        "Those are calculated based on modules function, not children.\n"
        "Check whether your model is as you desired (will be printed at the beginning of training).\n"
        "If unspecified, the spatial costs won't be applied to any layers (default behaviour).",
    )

    subparser.add_argument(
        "--type",
        required=True,
        choices=("linear", "convolution"),
        type=str.lower,
        help="""Type of layer. One of "Linear" or "Convolution" available."""
        "This option is case insensitive.",
    )

    subparser.add_argument(
        "--input",
        required=True,
        choices=("sequential", "concatenate", "mix"),
        type=str.lower,
        help="Type of input (how data will be presented for the neural net). Available modes: \n"
        """- "Sequential"\n- "Mix"\n- "Concatenate"\n"""
        "This option is case insensitive.",
    )

    subparser.add_argument(
        "--activation",
        required=True,
        help="torch.nn module to be used as network's activation.\nThis option is case sensitive "
        "and has to be specified EXACTLY as respective class inside 'torch.nn' package.",
    )

    subparser.add_argument(
        "--save", required=True, help="Where best model will be saved."
    )

    subparser.add_argument(
        "--tensorboard", required=True, help="Where tensorboard data will be saved."
    )

    subparser.add_argument(
        "--task",
        default=False,
        action="store_true",
        help="Type of Sampler used for sequential inputs.\n"
        "Either 'Random' (for random access across all tasks)"
        "or 'Task' (iterate over task sequentially as well).\n"
        "Only used when --input is chosen to be sequential and has to be specified in this case.\n"
        "Default: Random",
    )

    subparser.add_argument(
        "--proximity",
        required=False,
        default=0.0,
        type=float,
        help="Proximity loss parameter specified as float for all Spatial Layers (if any).\n "
        "Default: 0 (acts just like Linear)",
    )

    subparser.add_argument(
        "--regularization",
        required=False,
        default=0.0,
        type=float,
        help="Weight regularization coefficient.\n "
        "Default: 0 (no regularization)",
    )

    subparser.add_argument(
        "--transport",
        required=False,
        default=0.0,
        type=float,
        help="Transport loss parameter specified as float for all Spatial Layers (if any).\n "
        "Default: 0 (acts just like Linear)",
    )

    subparser.add_argument(
        "--norm",
        required=False,
        choices=("l1", "l2"),
        type=str.lower,
        help="Norm used in transport loss. Either L1 or L2. Case insensitive",
        default="l2",
    )

    subparser.add_argument(
        "--load",
        required=False,
        default=False,
        help="Path to the model you want to resume training from.",
    )


def record(subparsers) -> None:
    subparser = subparsers.add_parser(
        "record",
        help="Record activations of saved network.",
        formatter_class=argparse.RawTextHelpFormatter,
    )

    subparser.add_argument("--model", required=True, help="Path to trained model.")

    subparser.add_argument(
        "--input",
        required=True,
        choices=("sequential", "concatenate", "mix"),
        type=str.lower,
        help="Type of input (how data will be presented for the neural net). Available modes: \n"
        """- "Sequential"\n- "Mix"\n- "Concatenate"\n"""
        "This option is case insensitive.",
    )

    subparser.add_argument(
        "--datasets",
        required=False,
        default=["MNIST", "FashionMNIST", "KMNIST", "QMNIST"],
        nargs="+",
        help="Name of torchvision datasets used in experiment.\nRestrictions and traits:\n"
        "- Provided datasets have to be proper object from torchvision.datasets.\n"
        "- Provided datasets need the same input shape. \n"
        "- Provided datasets CAN HAVE varying number of labels (modulo will be taken).\n"
        "This option is case sensitive.\n"
        "Default: ['MNIST', 'FashionMNIST', 'KMNIST', 'QMNIST']",
    )

    subparser.add_argument(
        "--reduction",
        required=True,
        choices=("mean", "variance"),
        type=str.lower,
        help="What reduction to use. Available options:\n"
        """- "Mean"\n- "Variance"\n"""
        "This option is case insensitive.",
    )

    subparser.add_argument(
        "--save",
        required=True,
        help="Path where recorded data will be saved.\n"
        "Folder should be provided, structure will be created automatically.",
    )

    subparser.add_argument(
        "--task",
        default=False,
        action="store_true",
        help="Type of Sampler used for sequential inputs.\n"
        "Either 'Random' (for random access across all tasks)"
        "or 'Task' (iterate over task sequentially as well).\n"
        "Only used when --input is chosen to be sequential and has to be specified in this case.\n"
        "Default: Random",
    )

    subparser.add_argument(
        "--train",
        action="store_true",
        default=False,
        help="Whether to use training or validation dataset for plot generation. "
        "If specified, use training. Default: validation dataset.",
    )

    subparser.add_argument(
        "--root",
        required=False,
        default=tempfile.gettempdir(),
        help="Where downloaded datasets will be saved. By default inside your temporary folder.",
    )


def plot(subparsers) -> None:
    subparser = subparsers.add_parser("plot", help="Plot recorded activations.")
    subparser.add_argument(
        "--data",
        default=None,
        required=False,
        help="Folder where recorded activations from record step are stored. "
        "If specified, will plot per-task activations strength within every layer.",
    )

    subparser.add_argument(
        "--model",
        default=None,
        required=False,
        help="Path to model whose spatial parameters will be plotted.",
    )

    subparser.add_argument(
        "--save", required=True, help="Path where generated plots will be saved."
    )


def split(subparsers) -> None:
    subparser = subparsers.add_parser(
        "split",
        help="Split neural network into per-task networks.",
        formatter_class=argparse.RawTextHelpFormatter,
    )

    subparser.add_argument(
        "--method",
        required=True,
        choices=("activations", "greedy", "rescale", "probability"),
        type=str.lower,
        help="Method to split neural network.\n"
        "Available options:\n-'Activations' (to be fixed)\n-'Greedy'\n-'Rescale'\n-'Probability'\n"
        "Option is case insensitive.",
    )

    subparser.add_argument(
        "--where",
        required=False,
        default=None,
        type=int,
        nargs="+",
        help="Indices of linear modules that will be split.\n"
    )

    subparser.add_argument(
        "--data",
        default=None,
        required=False,
        help="Folder where recorded activations are stored (record step).\n"
        "Specify only if Activations used.",
    )

    subparser.add_argument("--model", required=True, help="Path to saved model.")

    subparser.add_argument(
        "--save",
        required=True,
        help="Path (folder) where generated models will be saved.",
    )


def score(subparsers) -> None:
    subparser = subparsers.add_parser(
        "score", help="Check performance of splitted neural networks."
    )

    subparser.add_argument(
        "--hyperparams",
        required=True,
        help="JSON file containing hyperparameters. "
        "Defaults probably shouldn't be changed as those are more related to hyperparameters search rather than experiments. "
        'If you want to find best model on given task, use shell wrapper "find" inside /src.',
    )

    subparser.add_argument(
        "--models",
        required=True,
        help="Path (folder) containing splitted models (from split step).",
    )

    subparser.add_argument(
        "--input",
        required=True,
        choices=("sequential", "concatenate", "mix"),
        type=str.lower,
        help="Type of input (how data will be presented for the neural net). Available modes: \n"
        """- "Sequential"\n- "Mix"\n- "Concatenate"\n"""
        "This option is case insensitive.",
    )

    subparser.add_argument(
        "--datasets",
        required=False,
        default=["MNIST", "FashionMNIST", "KMNIST", "QMNIST"],
        nargs="+",
        help=r"Name of torchvision datasets used in experiment. "
        "- Provided datasets have to be proper object from torchvision.datasets.\n"
        "- Provided datasets need the same input shape. \n"
        "- Provided datasets need CAN HAVE varying number of labels (modulo will be taken).\n"
        "This option is case sensitive.\n "
        """Default: ["MNIST", "FashionMNIST", "KMNIST", "QMNIST"]""",
    )

    subparser.add_argument(
        "--tensorboard",
        required=True,
        help="Where results will be saved to tensorboard",
    )

    subparser.add_argument(
        "--task",
        default=False,
        action="store_true",
        help="Type of Sampler used for sequential inputs.\n"
        "Either 'Random' (for random access across all tasks)"
        "or 'Task' (iterate over task sequentially as well).\n"
        "Only used when --input is chosen to be sequential and has to be specified in this case.\n"
        "Default: Random",
    )


    subparser.add_argument(
        "--train",
        action="store_true",
        default=False,
        help="Whether to use training or validation dataset to perform scoring.",
    )

    subparser.add_argument(
        "--root",
        required=False,
        default=tempfile.gettempdir(),
        help="Where downloaded datasets will be saved. By default inside your temporary folder.",
    )
