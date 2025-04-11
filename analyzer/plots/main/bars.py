from matplotlib import pyplot as plt

from analyzer.stats import calc_iv_var, calc_var_groups_stat, SH_InformationValueReport, SH_GroupsStatReport
from analyzer.preprocessing.preprocess import BinningParamsSingleVars, MapDictSingleVar
from analyzer.utils.general.types import DataFrame

from ..config import PlotConfig


def plot_var_stat(
        data: DataFrame, var_name: str, target_name: str = None,
        map_values: MapDictSingleVar = None, binning: BinningParamsSingleVars = True,
        del_values: list = None, xticks='name', annotation: bool = True, plot_config=None,
        print_points=True, return_table=True
):
    """
    План решения:
    1. Нужно как-то бинаризовать переменную, если это необходимо.
    2. В зависимости есть ли target или нет, получить таблицу с необходимой статистикой.
    3. Нарисовать график
    4. Отобразить табличку
    """

    if target_name is not None:
        report: SH_InformationValueReport.t = calc_iv_var(data, var_name, target_name, binning, map_values)
    else:
        report: SH_GroupsStatReport.t = calc_var_groups_stat(data, var_name, binning, map_values)

    if plot_config is None:
        plot_config = PlotConfig()

    with plt.style.context(plot_config.style):
        fig = plt.figure(figsize=plot_config.figure_size)
        ax1 = fig.add_subplot(211)
        xticks = xticks.title()
        palette = None
        if plot_config.color is None:
            palette = sns.color_palette(plot_config.colormap, data.shape[0]).as_hex()
            palette = reorder_palette(palette, data['Target Rate(%)' if target else 'Population(%)'])
        sns.barplot(x=xticks, y='Population(%)', ax=ax1, data=data, color=plot_config.color, palette=palette)

        if target:
            ax2 = ax1.twinx()
            sns.pointplot(x=xticks, y='Target Rate(%)', data=data, ax=ax2, color=plot_config.color2)
        else:
            ax2 = None

        if annotation:
            for i in range(data.shape[0]):
                x = data.loc[i, 'Number']
                pop = round(data.loc[i, 'Population(%)'], 1)
                rate = round(data.loc[i, 'Target Rate(%)'] if target else 0, 1)
                lb = f'Rate={rate}%\nPop-n={pop}%' if target else f'Pop-n={pop}%'
                y = pop + plot_config.annotation_delta
                ax1.text(x, y, lb, ha='center', size=plot_config.annotation_font_size)

        plot_config.appy_configs(fig, ax1, ax2)
        plt.show()

        if print_points and points:
            points = '[' + ', '.join([str(x) for x in points]) + ']'
            points = points.replace('inf', 'np.inf')
            print('\n\n', points, '\n\n')
        if return_table:
            del data['Number']
            return data