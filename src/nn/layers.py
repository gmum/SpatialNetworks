"""
This module should contain all versions of spatial layers.

All of them are input shape inferrable via `torchlayers` and created
at the end of module.

Also that's the reason why all the module have underscore before names;
underscored layers do not have shape inference capabilities.

"""

import torch

import torchlayers


# Check different initialization schemes and how it changes the outcome?
# Is it at all connected with popular initialization schemes like Xavier, would it work better?
class _SpatialLinear(torch.nn.Linear):
    """Standard linear layer with `positions` representing it's location in ND space.

    `N` sized vector will be used for each output neuron to represent
    it's location in `N`-dimensional space.

    Parameters
    ----------
    in_features : int
            Size of each input sample
    out_features : int
            Size of each output sample
    bias: bool, optional
            If set to `False`, the layer will not learn an additive bias.
            Default: `True`
    dim: int, optional
            Dimensionality of the layer. Default: `2` (2D spatial parameters)

    Returns
    -------
    type
            <return description>

    """

    def __init__(self, in_features, out_features, bias: bool = True, dim: int = 2):
        super().__init__(in_features, out_features, bias)
        self.positions = torch.nn.Parameter(torch.randn(dim, out_features))


# Somehow not working due to serialization error. Will fix soon in torchlayers
# Probably due to custom dimension inference
class _SpatialConv(torchlayers.convolution.Conv):
    """Standard convolution layer where channels have N-dimensional spatial positions.

    `N` sized vector will be used for each output channels to represent
    it's location in `N`-dimensional space.

    Based on input shape it either creates 1D, 2D or 3D convolution for inputs of shape
    3D, 4D, 5D respectively (including batch as first dimension).

    Additional `same` `padding` mode was added and set as default. Using it's input dimensions
    (except for channels) like height and width will be preserved (for odd kernel sizes).

    `kernel_size` got a default value of `3`.

    Otherwise acts exactly like PyTorch's Convolution, see
    `documentation <https://pytorch.org/docs/stable/nn.html#convolution-layers>`__.

    Parameters
    ----------
    in_channels: int
        Number of channels in the input image
    out_channels: int
        Number of channels produced by the convolution
    kernel_size: int or tuple, optional
        Size of the convolving kernel. Default: 3
    stride: int or tuple, optional
        Stride of the convolution. Default: 1
    padding: int or tuple, optional
        Zero-padding added to both sides of the input. Default: 0
    padding_mode: string, optional
        Accepted values `zeros` and `circular` Default: `zeros`
    dilation: int or tuple, optional
        Spacing between kernel elements. Default: 1
    groups: int, optional
        Number of blocked connections from input channels to output channels. Default: 1
    bias: bool, optional
        If ``True``, adds a learnable bias to the output. Default: ``True``
    dim: int, optional
            Dimensionality of the layer. Default: `2` (2D spatial parameters)

    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,  # It's usually 3, might want to add that as user specified argument
        stride=1,
        padding="same",  # This is kinda useful
        dilation=1,
        groups=1,
        bias=True,
        padding_mode="zeros",
        dim: int = 2,
    ):
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
            padding_mode,
        )
        self.positions = torch.nn.Parameter(torch.randn(dim, out_channels))


###############################################################################
#
#                   SHAPE INFERABLE VERSIONS OF LAYERS
#
###############################################################################


SpatialLinear = torchlayers.make_inferrable(_SpatialLinear)
SpatialConv = torchlayers.make_inferrable(_SpatialConv)


def spatial(module) -> bool:
    """Return True if module considered spatial.

    Parameters
    ----------
    module: torch.nn.Module
        PyTorch module

    Returns
    -------
    bool

    """
    return isinstance(module, (_SpatialLinear, _SpatialConv))
