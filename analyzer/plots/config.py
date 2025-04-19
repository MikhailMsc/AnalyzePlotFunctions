from dataclasses import dataclass, fields
from typing import Literal, Tuple, List, Union, Self

from matplotlib import pyplot as plt

AVAILABLE_STYLES = plt.style.available[:]


@dataclass
class PlotConfig:
    style: Literal[*AVAILABLE_STYLES] = None
    plot_size: Tuple[float, float] = None

    colormap: str = None
    color: str = None
    color2: str = None

    annotation_delta: float = 0
    annotation_font_size: float = 10

    title: str = None
    title_size: float = 18
    title_pad: float = 0.0

    xlabel: str = ''
    xlabel_size: float = 15
    xtick_size: float = 10
    x_rotation: float = None
    x_grid: bool = True
    xmax: float = None
    xmin: float = None

    ylabel: str = ''
    ylabel_size: float = 15
    ytick_size: float = 10
    y_rotation: float = None
    y_grid: bool = True
    ymax: float = None
    ymin: float = None

    y2label: str = ''
    y2_grid: bool = False
    y2max: float = None
    y2min: float = None

    x_to_top: bool = False
    x_inverse: bool = False
    y_to_right: bool = False
    y_inverse: bool = False

    columns_ratios: List[float] = None
    rows_ratios: List[float] = None
    empty_space: float = None
    grig_widths: float = None
    bar_width: float = 0.99
    side_grid: bool = None

    cbar_location: str = None
    cbar_width: float = 5
    cbar_pad: float = 1

    def merge(self, other) -> Self:
        self_fields = fields(self)
        for f in self_fields:
            if (val := getattr(other, f.name)) is not None:
                setattr(self, f.name, val)
        return self


def apply_plot_config(config: PlotConfig, fig, ax=None, ax2=None):
    if config.title:
        fig.suptitle(config.title, fontsize=config.title_size, y=0.98 + config.title_pad)
        # if side_axis:
        #     fig.suptitle(self.title, fontsize=self.title_size, y=0.98 + self.title_pad)
        # else:
        #     ax.set_title(self.title, fontdict={'fontsize': self.title_size}, y=self.title_pad)

    if ax is not None:
        if config.x_to_top:
            ax.xaxis.tick_top()
        if config.y_to_right:
            ax.yaxis.tick_right()
        if config.y_inverse:
            ax.invert_yaxis()
        if config.x_inverse:
            ax.invert_xaxis()

        ax.xaxis.label.set_fontsize(config.xlabel_size)
        ax.yaxis.label.set_fontsize(config.ylabel_size)

        for item in ax.get_xticklabels():
            item.set_fontsize(config.xtick_size)
        for item in ax.get_yticklabels():
            item.set_fontsize(config.ytick_size)

        if config.y_rotation is not None:
            ax.set_yticklabels(ax.yaxis.get_majorticklabels(), rotation=config.y_rotation)
        if config.x_rotation is not None:
            ax.set_xticklabels(ax.xaxis.get_majorticklabels(), rotation=config.x_rotation)

        if config.x_grid is not None:
            ax.xaxis.grid(config.x_grid)
        if config.y_grid is not None:
            ax.yaxis.grid(config.y_grid)

        if config.xlabel is not None:
            ax.set_xlabel(config.xlabel)

        if config.ylabel is not None:
            ax.set_ylabel(config.ylabel)

        if config.ymax is not None:
            ax.set_ylim(top=config.ymax)
        if config.ymin is not None:
            ax.set_ylim(bottom=config.ymin)

        if config.xmax is not None:
            ax.set_xlim(right=config.xmax)
        if config.xmin is not None:
            ax.set_xlim(left=config.xmin)

    if ax2:
        if config.y2_grid is not None:
            ax2.yaxis.grid(config.y2_grid)
        if config.y2label is not None:
            ax2.set_ylabel(config.y2label)

        ax2.yaxis.label.set_fontsize(config.ylabel_size)

        if config.color2 is not None:
            ax2.yaxis.label.set_color(config.color2)

        for item in ax2.get_yticklabels():
            item.set_fontsize(config.ytick_size)
            if config.color2 is not None:
                item.set_color(config.color2)

        if config.y2max is not None:
            ax2.set_ylim(top=config.y2max)
        if config.y2min is not None:
            ax2.set_ylim(bottom=config.y2min)
