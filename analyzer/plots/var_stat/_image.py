from copy import copy
from typing import List, Tuple

import seaborn as sns

from analyzer.utils.general.types import DataFrame
from analyzer.utils.framework_depends import get_shape

from ..config import PlotConfig


DEFAULT_PLOT_CONFIG = PlotConfig(
    plot_size=(12, 6),
    ylabel='Population(%)',
    title_pad=-0.03,
    y2label='Target Rate(%)',
    color2='red',
    colormap='Blues',
    xlabel_size=13.5,
    ylabel_size=13.5,
    style='fast'
)


def prepare_plot_config(report: DataFrame, plot_config: PlotConfig = None) -> Tuple[PlotConfig, List]:
    default_config = copy(DEFAULT_PLOT_CONFIG)
    if plot_config is None:
        plot_config = default_config
    else:
        plot_config = default_config.merge(plot_config)

    if plot_config.color is None:
        palette = sns.color_palette(plot_config.colormap, get_shape(report)[0]).as_hex()
    else:
        palette = []
    return plot_config, palette


def _reorder_palette(palette: List, order_by: List) -> List:
    assert len(palette) == len(order_by)

    import numpy as np
    order_colors = sorted(
        [(true_pos, value_pos) for value_pos, true_pos in enumerate(np.argsort(order_by))],
        key=lambda x: x[0]
    )
    palette = [palette[pos] for _, pos in order_colors]
    return palette


def apply_plot_config(config: PlotConfig, fig, ax1, ax2):
    if config.title:
        fig.suptitle(config.title, fontsize=config.title_size, y=0.98 + config.title_pad)

    ax1.xaxis.label.set_fontsize(config.xlabel_size)
    ax1.yaxis.label.set_fontsize(config.ylabel_size)

    for item in ax1.get_xticklabels():
        item.set_fontsize(config.xtick_size)
    for item in ax1.get_yticklabels():
        item.set_fontsize(config.ytick_size)

    if config.x_rotation is not None:
        ax1.set_xticklabels(ax1.xaxis.get_majorticklabels(), rotation=config.x_rotation)

    if config.x_grid is not None:
        ax1.xaxis.grid(config.x_grid)
    if config.y_grid is not None:
        ax1.yaxis.grid(config.y_grid)

    if config.xlabel:
        ax1.set_xlabel(config.xlabel)

    if config.ylabel:
        ax1.set_ylabel(config.ylabel)

    if config.ymax is not None:
        ax1.set_ylim(top=config.ymax)
    if config.ymin is not None:
        ax1.set_ylim(bottom=config.ymin)

    if config.xmax is not None:
        ax1.set_xlim(right=config.xmax)
    if config.xmin is not None:
        ax1.set_xlim(left=config.xmin)

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
