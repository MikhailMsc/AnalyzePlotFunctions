from typing import Union

import seaborn as sns

from analyzer.preprocessing import MapDictMultiVars, BinningParamsMultiVars
from analyzer.utils.general.types import DataFrame
from analyzer.stats import calc_concentration_report, SH_ConcentrationReport
from analyzer.utils.framework_depends import get_sub_df, set_column, drop_columns
from analyzer.utils.domain.columns import (
    C_POPULATION, C_TARGET_RATE, C_TARGET, C_TARGET_POPULATION, C_GROUP_IV,
    C_PARENT_MIN, C_PARENT_MAX, C_PARENT_MIN_TR, C_PARENT_MAX_TR
)

from ..config import PlotConfig
from ._order_values import get_order_vars_values
from ._utils import split_palette, DEFAULT_COLORMAP
from ._circles import plot_circles
from ._heatmap import plot_heatmap


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
    График отображения кросс-статистки по двум переменным.

    Args:
        data:                       Исследуемый датафрейм
        var_name_1:                 Название первой интересующей переменной
        var_name_2:                 Название второй интересующей переменной
        target_name:                Опционально. Название таргета
        map_values:                 Словарь для замены значений переменных (словарь, ключ = название переменной,
                                    значение = словарь старое-новое значение)
        binning:                    Параметры для биннинга
        histogram:                  Флаг, боковые диаграммы
        colorbar:                   Флаг, колорбар
        annotation:                 Флаг, аннотация графика
        plot_config:                Конфиг для графика
        min_population:             Минимальный размер популяции(%) для отображения на графике
        circles:                    Круговая диаграмма или прямоугольная
        _resize:                    Скрытый параметр. Подгон размера графика

    Returns:
        * - Опциональные колонки, только при наличии таргета.
        DataFrame:
            Var1_Name:                   Значения групп первой переменной
            Var2_Name:                   Значения групп второй переменной
            COUNT:                       Количество наблюдений в данном сегменте,
            POPULATION:                  Относительный размер сегмента,
            *TARGET:                     Количество таргетов в данном сегменте,
            *TARGET_RATE:                Target Rate в данном сегменте,
            *TARGET_POPULATION:          относительный размер таргета в данном сегменте,
            *GROUP_IV:                   information value данного сегмента,
            *PARENT_MIN:                 родительский сегмент с минимальным target_rate,
            *PARENT_MIN_TargetRate:      минимальный target_rate родительского сегмента,
            *PARENT_MAX:                 родительский сегмент с максимальным target_rate,
            *PARENT_MAX_TargetRate:      максимальный target_rate родительского сегмента

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
        _bin_by_target=has_target,
        _logging=False,
        _drop_single_vals=False
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
    plot_report = plot_report[plot_report[C_POPULATION.n] >= min_population].reset_index(drop=True)

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
    vars_cnt_values = {v: len(vals) for v, vals in order_vars_values.items()}
    if vars_cnt_values[var_name_1] <= vars_cnt_values[var_name_2]:
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
        plot_circles(
            var_name_1, var_name_2, colorbar, histogram,
            plot_config, population_stats, target_stats,
            hist_report, palette_x_top, palette_y_right,
            order_vars_values, labels
        )
    else:
        plot_heatmap(
            var_name_1, var_name_2, colorbar, histogram,
            plot_config, target_stats if has_target else population_stats,
            hist_report, labels, palette_main, palette_x_top, palette_y_right,
            _resize
        )

    var_name_1 = plot_report['Var1_Name'][0]
    var_name_2 = plot_report['Var2_Name'][0]

    map_dict = {var: {val: i for i, val in enumerate(vals)} for var, vals in order_vars_values.items()}
    plot_report['Var1_Value_Order'] = plot_report['Var1_Value'].apply(lambda x: map_dict[var_name_1][x])
    plot_report['Var2_Value_Order'] = plot_report['Var2_Value'].apply(lambda x: map_dict[var_name_2][x])
    plot_report = plot_report.sort_values(['Var1_Value_Order', 'Var2_Value_Order']).reset_index(drop=True)
    plot_report.rename(columns={'Var1_Value': var_name_1, 'Var2_Value': var_name_2}, inplace=True)
    plot_report.drop(columns=['Var1_Name', 'Var2_Name','Var1_Value_Order', 'Var2_Value_Order'], inplace=True)

    return plot_report


def _prepare_labels(pop_stat, tr_stat):
    import pandas as pd
    labels = pop_stat.map(lambda x: None if pd.isnull(x) else f'Pop-n = {x:.1f}%')

    if tr_stat is not None:
        tr_stat = tr_stat.map(lambda x: None if pd.isnull(x) else f'TRate = {x:.1f}%')
        labels = labels + '\n' + tr_stat
    return labels.to_numpy()


