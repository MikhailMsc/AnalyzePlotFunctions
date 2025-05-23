from typing import Union

from matplotlib import pyplot as plt
import seaborn as sns

from analyzer.stats import calc_iv_var, calc_var_groups_stat, SH_InformationValueReport, SH_GroupsStatReport
from analyzer.preprocessing.preprocess import BinningParamsSingleVars, MapDictSingleVar
from analyzer.utils.general.types import DataFrame
from analyzer.utils.framework_depends import get_shape, get_series_from_df, series_to_list
from analyzer.utils.domain.columns import C_POPULATION, C_TARGET_RATE, C_GROUP, C_GROUP_NUMBER

from ..config import PlotConfig
from ._image import prepare_plot_config, apply_plot_config


def plot_var_stat(
        data: DataFrame, var_name: str, target_name: str = None,
        map_values: MapDictSingleVar = None, binning: BinningParamsSingleVars = True,
        annotation: bool = True, plot_config: PlotConfig = None,
        _mark_bar=None,
) -> Union[SH_InformationValueReport.t, SH_GroupsStatReport.t]:
    """
    График для отображения статистики по переменной.

    Args:
        data:               Исследуемый датафрейм
        var_name:           Название интересующей переменной
        target_name:        Опционально. Название таргета
        map_values:         Словарь для замены значений переменных (словарь старое-новое значение)
        binning:            Параметры для биннинга
        annotation:         Аннотация на графике
        plot_config:        Конфиг для графика
        _mark_bar:          Скрытый параметр. Название категории, столбце которой необходимо выделить.

    Returns:
        * - Опциональные колонки, есть только при наличии таргета.
        DataFrame:
            VARNAME:                имя переменной
            GROUP_NUMBER:           номер категории
            GROUP:                  значение категории
            COUNT:                  общий размер категории
            *TARGET:                количество таргетов в данной категории
            POPULATION:             относительный размер категории
            *TARGET_POPULATION:     относительный размер таргета в данной категории
            *TARGET_RATE:           Target Rate в данной категории
            *GROUP_IV:              information value данной категории
            *TOTAL_IV:              information value всей переменной
    """

    has_target = target_name is not None
    if has_target:
        report: SH_InformationValueReport.t = calc_iv_var(data, var_name, target_name, map_values, binning)
    else:
        report: SH_GroupsStatReport.t = calc_var_groups_stat(data, var_name, map_values, binning)

    plot_config, palette = prepare_plot_config(report, plot_config)

    with plt.style.context(plot_config.style):
        fig = plt.figure(figsize=plot_config.plot_size)
        ax1 = fig.add_subplot(111)

        x = series_to_list(get_series_from_df(report, C_GROUP.n))
        y = series_to_list(get_series_from_df(report, C_POPULATION.n))

        plot_config.ymax = max(y) * 1.15
        plot_config.annotation_delta = max(y) * 0.03

        if has_target:
            z = series_to_list(get_series_from_df(report, C_TARGET_RATE.n))
        else:
            z = y

        if _mark_bar is None:
            if plot_config.color is not None:
                sns.barplot(
                    x=x, y=y,
                    ax=ax1, data=report, color=plot_config.color,
                    legend=False
                )
            else:
                sns.barplot(
                    x=x, y=y,
                    ax=ax1, data=report, color=plot_config.color, hue=z,
                    palette=palette or None,
                    legend=False
                )
        else:
            palette = sns.color_palette('Greys', 100).as_hex()
            marked_bar = [v == _mark_bar for v in x]

            sns.barplot(
                x=x, y=y,
                ax=ax1, data=report, hue=marked_bar,
                palette=[palette[10], palette[-1]],
                legend=False
            )

        if has_target:
            y2 = series_to_list(get_series_from_df(report, C_TARGET_RATE.n))
            ax2 = ax1.twinx()
            sns.pointplot(x=x, y=y2, data=report, ax=ax2, color=plot_config.color2)
        else:
            ax2 = None

        if annotation:
            for i in range(get_shape(report)[0]):
                pop = round(report[C_POPULATION.n][i], 1)
                if has_target:
                    rate = round(report[C_TARGET_RATE.n][i], 1)
                    lb = f'TRate={rate}%\nPop-n={pop}%'
                else:
                    lb = f'Pop-n={pop}%'
                y = pop + plot_config.annotation_delta
                ax1.text(i, y, lb, ha='center', size=plot_config.annotation_font_size)

        apply_plot_config(plot_config, fig, ax1, ax2)
        plt.show()
        report.drop(columns=[C_GROUP_NUMBER.n], inplace=True)
        return report
