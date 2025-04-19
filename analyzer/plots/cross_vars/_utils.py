from typing import Tuple

from matplotlib import pyplot as plt, gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable

from ..config import PlotConfig

DEFAULT_COLORMAP = 'Blues'


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


def resize_plot(plot_config, shape):
    y_size = min(plot_config.plot_size)
    x_size = (y_size / shape[0]) * shape[1]

    plot_config.plot_size = (x_size, y_size)


def get_all_axes(plot_config: PlotConfig, colorbar: bool, histogram: bool, shape: Tuple[int, int]):
    fig = plt.figure(figsize=plot_config.plot_size)
    if histogram:
        rows = 4 if colorbar else 2
        grid = gridspec.GridSpec(
            nrows=rows, ncols=2,
            width_ratios=plot_config.columns_ratios,
            height_ratios=plot_config.rows_ratios,
            wspace=plot_config.empty_space,
            hspace=plot_config.empty_space *
                   (shape[1] / shape[0]) *
                   (plot_config.columns_ratios[0] / plot_config.rows_ratios[1])
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


def set_grid_proportions(plot_config: PlotConfig, histogram: bool, colorbar: bool, shape):
    y_size = min(plot_config.plot_size)

    if not histogram:
        if not colorbar:
            x_size = y_size * shape[1] / shape[0]
        else:
            main_x = y_size * shape[1] / shape[0]
            x_size = main_x / (1 - plot_config.cbar_width / 100)
    else:
        if colorbar:
            plot_config.rows_ratios = [0.12, 0.70, 0.12, 0.06]
        else:
            plot_config.rows_ratios = [0.15, 0.85]

        assert sum(plot_config.rows_ratios) == 1
        main_y = y_size * plot_config.rows_ratios[1]
        hist_size = y_size * plot_config.rows_ratios[0]

        main_x = main_y * shape[1] / shape[0]
        x_size = main_x + hist_size

        hist_ratio = hist_size / x_size
        plot_config.columns_ratios = [1 - hist_ratio, hist_ratio]

    plot_config.plot_size = (x_size, y_size)
