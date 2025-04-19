from typing import Union

from matplotlib import pyplot as plt
import seaborn as sns

from analyzer.stats import calc_iv_var, calc_var_groups_stat, SH_InformationValueReport, SH_GroupsStatReport
from analyzer.preprocessing.preprocess import BinningParamsSingleVars, MapDictSingleVar
from analyzer.utils.general.types import DataFrame
from analyzer.utils.framework_depends import get_shape, get_series_from_df, series_to_list
from analyzer.utils.domain.columns import C_POPULATION, C_TARGET_RATE, C_GROUP

from ..config import PlotConfig, apply_plot_config
from ._image import prepare_plot_config


def plot_var_stat(
        data: DataFrame, var_name: str, target_name: str = None,
        map_values: MapDictSingleVar = None, binning: BinningParamsSingleVars = True,
        xlabel='Group', annotation: bool = True, plot_config: PlotConfig = None
) -> Union[SH_InformationValueReport.t, SH_GroupsStatReport.t]:

    has_target = target_name is not None
    if has_target:
        report: SH_InformationValueReport.t = calc_iv_var(data, var_name, target_name, binning, map_values)
    else:
        report: SH_GroupsStatReport.t = calc_var_groups_stat(data, var_name, binning, map_values)

    plot_config, palette = prepare_plot_config(report, xlabel, has_target, plot_config)

    with plt.style.context(plot_config.style):
        fig = plt.figure(figsize=plot_config.plot_size)
        ax1 = fig.add_subplot(111)

        x = series_to_list(get_series_from_df(report, C_GROUP.n))
        y = series_to_list(get_series_from_df(report, C_POPULATION.n))
        plot_config.ymax = max(y) + 7

        if has_target:
            z = series_to_list(get_series_from_df(report, C_TARGET_RATE.n))
        else:
            z = y

        sns.barplot(
            x=x, y=y,
            ax=ax1, data=report, color=plot_config.color, hue=z,
            palette=palette or None, legend=False
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
        return report
