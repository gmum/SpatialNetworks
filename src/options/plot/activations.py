import builtins
import dataclasses
import pathlib
import typing

import matplotlib.pyplot as plt
import numpy as np
import torch


@dataclasses.dataclass(init=True)
class DenseActivations:
    ax: typing.Any
    colormap: typing.Any = None
    horizontal: bool = False
    value_max: int = None
    value_min: int = None
    neuron_size: int = None
    edge_colors: typing.Any = None
    orientation: str = "vertical"

    def __post_init__(self):
        self.ax.axis("off")

    def __call__(self, *layers: torch.Tensor) -> None:
        def _find_extremes(layers, initial_value, function: str):
            return (
                getattr(builtins, function)(map(getattr(torch, function), layers))
                if initial_value is None
                else initial_value
            )

        vmin, vmax = (
            _find_extremes(layers, self.value_min, "min"),
            _find_extremes(layers, self.value_max, "max"),
        )
        longest_layer = max(map(len, layers))

        last_scatter = None
        for index, layer in enumerate(layers):
            spacing = int((longest_layer - len(layer)) * 0.5)
            layer_position_in_model = np.full(len(layer), index)
            neuron_positions = range(spacing, spacing + len(layer))
            last_scatter = self.ax.scatter(
                x=neuron_positions if self.horizontal else layer_position_in_model,
                y=layer_position_in_model if self.horizontal else neuron_positions,
                c=layer.float().numpy(),
                cmap=self.colormap,
                s=self.neuron_size,
                vmin=vmin,
                vmax=vmax,
                edgecolors=self.edge_colors,
            )

        plt.colorbar(last_scatter, ax=self.ax, orientation=self.orientation)


def plot(data, args):
    folder = pathlib.Path(args.save)
    folder.mkdir(parents=True, exist_ok=True)
    for index, task in enumerate(data):
        fig = plt.figure(figsize=(12, 12))
        ax = fig.gca()
        plotter = DenseActivations(ax=ax)
        plotter(*task)
        fig.savefig(folder / f"task_{index}.png")
