from copy import copy
from typing import Tuple, Union, Dict

import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt

from analyzer.utils.domain.columns import C_POPULATION
from analyzer.utils.general.types import DataFrame

from ..config import PlotConfig
from ._utils import DEFAULT_COLORMAP, resize_plot, get_all_axes, set_grid_proportions


def plot_heatmap(
        var_name_1: str, var_name_2: str, colorbar: bool, histogram: bool,
        plot_config: Union[PlotConfig, None],
        plot_stats: DataFrame, hist_report: Dict[str, DataFrame],
        labels: DataFrame, palette_main, palette_x_top, palette_y_right,
        resize: bool,
):
    plot_config = _prepare_plot_config(
        var_name_1, var_name_2, colorbar, histogram, plot_stats.shape, resize, plot_config,
    )

    with plt.style.context(plot_config.style):
        fig, ax_main, ax_hist_top, ax_hist_right, ax_colorbar = get_all_axes(
            plot_config, colorbar, histogram, plot_stats.shape
        )
        cross_plot = sns.heatmap(
            plot_stats,
            cmap=palette_main,
            cbar=False, linewidths=plot_config.grid_widths,
            annot=labels, annot_kws={'size': plot_config.annotation_font_size}, fmt='',
            ax=ax_main
        )
        cross_plot.invert_yaxis()

        if histogram:
            ax_hist_right.barh(
                y=np.arange(hist_report[var_name_2].shape[0]) + 0.5,
                height=plot_config.bar_width,
                width=hist_report[var_name_2][C_POPULATION.n],
                align='center', color=palette_y_right
            )
            plt.setp(ax_hist_right.get_yticklabels(), visible=False)

            ax_hist_top.bar(
                x=np.arange(hist_report[var_name_1].shape[0]) + 0.5,
                width=plot_config.bar_width,
                height=hist_report[var_name_1][C_POPULATION.n],
                align='center', color=palette_x_top
            )
            plt.setp(ax_hist_top.get_xticklabels(), visible=False)

        if colorbar:
            if plot_config.cbar_location in ['right', 'left']:
                cbar_orientation = 'vertical'
            else:
                cbar_orientation = 'horizontal'
            plt.colorbar(ax_main.get_children()[0], cax=ax_colorbar, orientation=cbar_orientation)

        _apply_plot_config(plot_config, fig, ax_main, ax_hist_top, ax_hist_right, ax_colorbar)
        plt.show()


_DEFAULT_PLOT_CONFIG_NO_HIST = PlotConfig(
    plot_size=(8, 8),
    y_rotation=0,
    x_rotation=20,
    colormap=DEFAULT_COLORMAP,
    cbar_location='right',
    bar_width=0.99,
    empty_space=0.04,
    annotation_font_size=10,
    cbar_pad=3,
    style='fast',
    side_grid=False,
)

_DEFAULT_PLOT_CONFIG_WITH_HIST = copy(_DEFAULT_PLOT_CONFIG_NO_HIST)
_DEFAULT_PLOT_CONFIG_WITH_HIST.plot_size = (9, 9)

_DEFAULT_PLOT_CONFIG_WITH_HIST_CBAR = copy(_DEFAULT_PLOT_CONFIG_NO_HIST)
_DEFAULT_PLOT_CONFIG_WITH_HIST_CBAR.plot_size = (10, 10)
_DEFAULT_PLOT_CONFIG_WITH_HIST_CBAR.cbar_location = 'bottom'


def _prepare_plot_config(
        xlabel: str, ylabel: str,
        colorbar: bool, histogram: bool,
        shape: Tuple[int, int], resize: bool,
        plot_config: PlotConfig = None,

) -> PlotConfig:
    if not histogram:
        default_config = copy(_DEFAULT_PLOT_CONFIG_NO_HIST)
    elif colorbar:
        default_config = copy(_DEFAULT_PLOT_CONFIG_WITH_HIST_CBAR)
    else:
        default_config = copy(_DEFAULT_PLOT_CONFIG_WITH_HIST)

    if plot_config is None:
        plot_config = default_config
    else:
        plot_config = default_config.merge(plot_config)

    plot_config.xlabel = xlabel.title()
    plot_config.ylabel = ylabel.title()
    if resize:
        resize_plot(plot_config, shape)

    set_grid_proportions(plot_config, histogram, colorbar, shape)
    return plot_config


def _apply_plot_config(config: PlotConfig, fig, ax_main, ax_hist_top, ax_hist_right, ax_colorbar):
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

    if config.side_grid is not None and ax_hist_top is not None:
        ax_hist_top.xaxis.grid(config.side_grid)
        ax_hist_top.yaxis.grid(config.side_grid)

        ax_hist_right.xaxis.grid(config.side_grid)
        ax_hist_right.yaxis.grid(config.side_grid)



