from typing import Union, Dict, List

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.collections import PatchCollection
import seaborn as sns

from analyzer.preprocessing import MapDictMultiVars, BinningParamsMultiVars, preprocess_df
from analyzer.utils.general.types import DataFrame
from analyzer.stats import calc_concentration_report, SH_ConcentrationReport
from analyzer.utils.framework_depends import get_sub_df, set_column, drop_columns
from analyzer.utils.domain.columns import (
    C_POPULATION, C_TARGET_RATE, C_TARGET, C_TARGET_POPULATION, C_GROUP_IV,
    C_PARENT_MIN, C_PARENT_MAX, C_PARENT_MIN_TR, C_PARENT_MAX_TR
)

from ._order_values import get_order_vars_values
from ._image import split_palette, prepare_plot_config_heatmap, get_all_axes, apply_plot_config_heatmap, \
    DEFAULT_COLORMAP
from ..config import PlotConfig


def plot_cross_vars(
        data: DataFrame, var_name_1: str, var_name_2: str, target_name: str = None,
        map_values: MapDictMultiVars = None, binning: BinningParamsMultiVars = True,
        histogram: bool = True, colorbar: bool = True,
        annotation: bool = True, plot_config: PlotConfig = None,
        min_population: float = 0,
        circles: bool = False,
        _resize: bool = True

) -> Union[SH_ConcentrationReport.t]:
    """
    План решения:
    1. В зависимости есть ли target или нет, получить таблицу с необходимой статистикой.
    2. Нарисовать график
    3. Отобразить табличку
    """
    has_target = target_name is not None
    if not has_target:
        data = get_sub_df(data, [var_name_1, var_name_2])
        target_name = '__target__'
        data = set_column(data, 1, '__target__')

    report: SH_ConcentrationReport.t = calc_concentration_report(
        data, target_name, combo_min=1, combo_max=2,
        analyze_vars=[var_name_1, var_name_2],
        binning=binning,
        map_values=map_values,
        _tqdm=False,
        _validate_target=has_target,
        _bin_by_target=has_target
    )

    if not has_target:
        report = drop_columns(
            report,
            [
                C_TARGET.n, C_TARGET_RATE.n, C_TARGET_POPULATION.n, C_GROUP_IV.n,
                C_PARENT_MIN.n, C_PARENT_MIN_TR.n, C_PARENT_MAX.n, C_PARENT_MAX_TR.n
            ]
        )

    plot_report = report[~report['Var2_Name'].isnull()]
    plot_report = plot_report[plot_report[C_POPULATION.n] >= min_population]

    hist_report = report[report['Var2_Name'].isnull()]
    _hist_report = dict()
    order_vars_values = dict()
    for i, var in enumerate(sorted([var_name_1, var_name_2])):
        sub_report = hist_report[hist_report['Var1_Name'] == var]
        var_values = plot_report.loc[plot_report[f'Var{i+1}_Name'] == var, f'Var{i+1}_Value'].to_list()
        sub_report = sub_report[sub_report['Var1_Value'].isin(var_values)]
        sub_report.reset_index(drop=True, inplace=True)
        order_vars_values[var] = get_order_vars_values(sub_report, has_target)
        sub_report.set_index('Var1_Value', inplace=True)
        sub_report = sub_report.loc[order_vars_values[var]].reset_index(drop=True)
        _hist_report[var] = sub_report

    hist_report = _hist_report
    vars_cnt_values = {v: len(v) for v, vals in order_vars_values.items()}
    if vars_cnt_values[var_name_1] > vars_cnt_values[var_name_2]:
        var_name_1, var_name_2 = var_name_2, var_name_1

    var_value_x, var_value_y = (
        ('Var2_Value', 'Var1_Value')
        if var_name_1 > var_name_2 else
        ('Var1_Value', 'Var2_Value')
    )

    target_stats = None
    if has_target:
        target_stats = plot_report.pivot(index=var_value_y, columns=var_value_x, values=C_TARGET_RATE.n)
        target_stats = target_stats.loc[order_vars_values[var_name_2], order_vars_values[var_name_1]]

    population_stats = plot_report.pivot(index=var_value_y, columns=var_value_x, values=C_POPULATION.n)
    population_stats = population_stats.loc[order_vars_values[var_name_2], order_vars_values[var_name_1]]

    labels = False
    if annotation:
        labels = _prepare_labels(population_stats, target_stats)

    cnt_colors = population_stats.shape[0] * population_stats.shape[1]
    if plot_config is None or plot_config.colormap is None:
        colormap = DEFAULT_COLORMAP
    else:
        colormap = plot_config.colormap

    if histogram:
        cnt_colors += population_stats.shape[0] + population_stats.shape[1]
        palette = sns.color_palette(colormap or DEFAULT_COLORMAP, cnt_colors).as_hex()

        value_column = C_TARGET_RATE.n if has_target else C_POPULATION.n
        order_values = (
                hist_report[var_name_1][value_column].to_list() +
                list((target_stats if has_target else population_stats).values.flatten()) +
                hist_report[var_name_2][value_column].to_list()
        )

        palette_x_top, palette_y_right, palette_main = split_palette(
            palette, order_values,
            cnt_1=hist_report[var_name_1].shape[0],
            cnt_2=hist_report[var_name_2].shape[0]
        )

    else:
        palette_main = sns.color_palette(colormap, cnt_colors).as_hex()
        palette_x_top = palette_y_right = None

    if circles:
        raise NotImplementedError
    else:
        _plot_heatmap(
            var_name_1, var_name_2, colorbar, histogram,
            plot_config, target_stats if has_target else population_stats,
            hist_report, labels, palette_main, palette_x_top, palette_y_right,
            _resize
        )
    return plot_report


def _prepare_labels(pop_stat, tr_stat):
    import pandas as pd
    labels = pop_stat.map(lambda x: None if pd.isnull(x) else f'Pop-n = {x:.1f}%')

    if tr_stat is not None:
        tr_stat = tr_stat.map(lambda x: None if pd.isnull(x) else f'TRate = {x:.1f}%')
        labels = labels + '\n' + tr_stat
    return labels.to_numpy()


def _plot_heatmap(
        var_name_1: str, var_name_2: str, colorbar: bool, histogram: bool,
        plot_config: Union[PlotConfig, None],
        plot_stats: DataFrame, hist_report: Dict[str, DataFrame],
        labels: DataFrame, palette_main, palette_x_top, palette_y_right,
        resize: bool,
):
    plot_config = prepare_plot_config_heatmap(
        var_name_1, var_name_2, colorbar, histogram, plot_stats.shape, resize, plot_config,

    )

    with plt.style.context(plot_config.style):
        fig, ax_main, ax_hist_top, ax_hist_right, ax_colorbar = get_all_axes(plot_config, colorbar, histogram)
        cross_plot = sns.heatmap(
            plot_stats,
            cmap=palette_main,
            cbar=False, linewidths=plot_config.grig_widths,
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

        apply_plot_config_heatmap(plot_config, fig, ax_main, ax_hist_top, ax_hist_right, ax_colorbar)
        plt.show()


def _plot_circles(
    var_name_1: str, var_name_2: str, colorbar: bool, histogram: bool,
    plot_config: Union[PlotConfig, None], population_stats: DataFrame,
    target_stats: Union[DataFrame, None], hist_report: Dict[str, DataFrame],
    order_vars_values: Dict[str, List], labels: DataFrame
):
    plot_config = prepare_plot_config_heatmap(
        var_name_1, var_name_2, colorbar, histogram, plot_config
    )
    has_target = target_stats is not None
    value_column = C_TARGET_RATE.n if has_target else C_POPULATION.n

    with plt.style.context(plot_config.style):
        fig, ax_main, ax_hist_top, ax_hist_right, ax_colorbar = get_all_axes(plot_config, colorbar, histogram)

        x, y = np.meshgrid(order_vars_values[var_name_1], order_vars_values[var_name_2])
        radius = (population_stats / population_stats.max().max()) / 2
        circles = [
            (
                plt.Circle((x, y), radius=radius.loc[y_lbl, x_lbl]),
                (target_stats if has_target else population_stats).loc[y_lbl, x_lbl]
            )
            for x, x_lbl in enumerate(order_vars_values[var_name_1])
            for y, y_lbl in enumerate(order_vars_values[var_name_2])
        ]
        circles, rate = zip(*circles)
        col = PatchCollection(circles, array=rate, cmap=plot_config.colormap)
        ax_main.add_collection(col)

        cnt_x_values = len(order_vars_values[var_name_1])
        cnt_y_values = len(order_vars_values[var_name_2])
        ax_main.set(
            xticks=np.arange(cnt_x_values),
            yticks=np.arange(cnt_y_values),
            xticklabels=order_vars_values[var_name_1],
            yticklabels=order_vars_values[var_name_2]
        )
        ax_main.set_xticks(np.arange(cnt_x_values + 1) - 0.5, minor=True)
        ax_main.set_yticks(np.arange(cnt_y_values + 1) - 0.5, minor=True)

        for y_i in range(cnt_y_values):
            for x_i in range(cnt_x_values):
                txt = labels[y_i, x_i]
                if txt is np.nan:
                    txt = ''
                ax_main.annotate(
                    txt, xy=(x_i, y_i + 0.25),
                    fontsize=plot_config.annotation_font_size,
                    horizontalalignment='center'
                )

        ax_main.grid(which='minor')