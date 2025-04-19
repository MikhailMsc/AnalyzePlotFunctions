from copy import copy
from typing import Tuple

from matplotlib import pyplot as plt, gridspec
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable

from ..config import PlotConfig


def split_palette(palette, order_by, cnt_1, cnt_2):
    import numpy as np
    order_colors = sorted(
        [(true_pos, value_pos) for value_pos, true_pos in enumerate(np.argsort(order_by))],
        key=lambda x: x[0]
    )
    palette_1 = [palette[pos] for _, pos in order_colors[:cnt_1]]
    palette_2 = [palette[pos] for _, pos in order_colors[-cnt_2:]]
    remain_palette = [value for value in palette if value not in palette_1 + palette_2]
    return palette_1, palette_2, remain_palette


DEFAULT_COLORMAP = 'Blues'


_DEFAULT_PLOT_CONFIG_NO_HIST = PlotConfig(
    plot_size=(9, 9),
    y_rotation=0,
    x_rotation=20,
    colormap=DEFAULT_COLORMAP,
    cbar_location='right',
    bar_width=0.99,
    empty_space=0.2,
    annotation_font_size=8,
    cbar_pad=3,
    style='fast',
    side_grid=False
)

_DEFAULT_PLOT_CONFIG_WITH_CBAR = copy(_DEFAULT_PLOT_CONFIG_NO_HIST)
_DEFAULT_PLOT_CONFIG_WITH_CBAR.rows_ratios = [1, 6, 0.5, 0.5]
_DEFAULT_PLOT_CONFIG_WITH_CBAR.columns_ratios = [6.2, 0.6]
_DEFAULT_PLOT_CONFIG_WITH_CBAR.cbar_location = 'bottom'

_DEFAULT_PLOT_CONFIG_NO_CBAR = copy(_DEFAULT_PLOT_CONFIG_WITH_CBAR)
_DEFAULT_PLOT_CONFIG_NO_CBAR.rows_ratios = [1.3, 6]


def prepare_plot_config_heatmap(
        xlabel: str, ylabel: str,
        colorbar: bool, histogram: bool,
        shape: Tuple[int, int], resize: bool,
        plot_config: PlotConfig = None,

) -> PlotConfig:
    if not histogram:
        default_config = copy(_DEFAULT_PLOT_CONFIG_NO_HIST)
    elif colorbar:
        default_config = copy(_DEFAULT_PLOT_CONFIG_WITH_CBAR)
    else:
        default_config = copy(_DEFAULT_PLOT_CONFIG_NO_CBAR)

    if plot_config is None:
        plot_config = default_config
    else:
        plot_config = default_config.merge(plot_config)

    plot_config.xlabel = xlabel.title()
    plot_config.ylabel = ylabel.title()
    if resize:
        _resize_plot(plot_config, shape)
    return plot_config


def get_all_axes(plot_config: PlotConfig, colorbar: bool, histogram: bool):
    fig = plt.figure(figsize=plot_config.plot_size)

    if histogram:
        rows = 4 if colorbar else 2
        grid = gridspec.GridSpec(
            nrows=rows, ncols=2,
            width_ratios=plot_config.columns_ratios,
            height_ratios=plot_config.rows_ratios,
            wspace=plot_config.empty_space - 0.06,
            hspace=plot_config.empty_space
        )
        ax_main = fig.add_subplot(grid[1, 0])
        ax_hist_top = fig.add_subplot(grid[0, 0], sharex=ax_main)
        ax_hist_right = fig.add_subplot(grid[1, 1], sharey=ax_main)
        if colorbar:
            ax_colorbar = fig.add_subplot(grid[3, 0])
        else:
            ax_colorbar = None
    else:
        ax_main = fig.add_subplot(111)
        ax_hist_top = ax_hist_right = None

        if colorbar:
            ax_divider = make_axes_locatable(ax_main)
            ax_colorbar = ax_divider.append_axes(
                plot_config.cbar_location,
                size='{}%'.format(plot_config.cbar_width),
                pad='{}%'.format(plot_config.cbar_pad)
            )
        else:
            ax_colorbar = None

    return fig, ax_main, ax_hist_top, ax_hist_right, ax_colorbar


def _resize_plot(plot_config, shape):
    y_size = min(plot_config.plot_size)
    x_size = (y_size / shape[0]) * shape[1]

    plot_config.plot_size = (x_size, y_size)


def apply_plot_config_heatmap(config: PlotConfig, fig, ax_main, ax_hist_top, ax_hist_right, ax_colorbar):
    if config.title:
        fig.suptitle(config.title, fontsize=config.title_size, y=0.98 + config.title_pad)

    ax_main.xaxis.label.set_fontsize(config.xlabel_size)
    ax_main.yaxis.label.set_fontsize(config.ylabel_size)

    for item in ax_main.get_xticklabels():
        item.set_fontsize(config.xtick_size)
    for item in ax_main.get_yticklabels():
        item.set_fontsize(config.ytick_size)

    if config.y_rotation is not None:
        ax_main.set_yticklabels(ax_main.yaxis.get_majorticklabels(), rotation=config.y_rotation)
    if config.x_rotation is not None:
        ax_main.set_xticklabels(ax_main.xaxis.get_majorticklabels(), rotation=config.x_rotation)

    if config.xlabel is not None:
        ax_main.set_xlabel(config.xlabel)

    if config.ylabel is not None:
        ax_main.set_ylabel(config.ylabel)

    if config.side_grid is not None:
        ax_hist_top.xaxis.grid(config.side_grid)
        ax_hist_top.yaxis.grid(config.side_grid)

        ax_hist_right.xaxis.grid(config.side_grid)
        ax_hist_right.yaxis.grid(config.side_grid)
