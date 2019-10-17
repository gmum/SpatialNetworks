import torch

import torchlayers


# Check different initialization schemes and how it changes the outcome?
# Is it at all connected with popular initialization schemes like Xavier, would it work better?
class _SpatialLinear(torch.nn.Linear):
    def __init__(self, in_features, out_features, bias: bool = True):
        super().__init__(in_features, out_features, bias)
        self.positions = torch.nn.Parameter(torch.randn(2, out_features))


# Somehow not working due to serialization error. Will fix soon in torchlayers
# Probably due to custom dimension inference
class _SpatialConv(torchlayers.convolution.Conv):
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
        self.positions = torch.nn.Parameter(torch.randn(2, out_channels))


# Make the above shape inferrable, makes writing networks easier
SpatialLinear = torchlayers.make_inferrable(_SpatialLinear)
SpatialConv = torchlayers.make_inferrable(_SpatialConv)
