from config import BASE_CONFIG
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
import seaborn as sns
from utils.framework_depends import reorder_palette
from binning import bin_variable


def var_stat(data: pd.DataFrame, var: str, target: str = None, binning: BinningParamsSingleVars = True, del_values: list = None,
             plot=True, xticks='name', annotation: bool = True, plot_config=None,
             print_points=True, return_table=True):
    """
    Parameters
    ----------
    data
    var
    target
    binargs
    del_values
    plot
    xticks
    annotation
    plot_config
    print_points
    return_table

    Returns
    -------

    """
    base_config = BASE_CONFIG.__copy__()
    base_config.title = var.title()
    base_config.xlabel = 'Group'
    base_config.ylabel = 'Population(%)'
    base_config.y2label = 'Target Rate(%)'
    base_config.colormap = 'GnBu'
    base_config.annotation_delta = 1
    plot_config = base_config if plot_config is None else base_config.merge_other_config(plot_config)

    data = data[[var, target] if target else [var]]
    if str(data[var].dtype) in ['category', 'object']:
        points, data['bin'] = None, data[var]
        binned = False
    else:
        binargs = {
            **{'target_name': data[target] if target else None, 'return_points': True},
            **binargs}
        points, data['bin'] = bin_variable(data[var], **binargs)
        binned = True

    if del_values:
        del_values = [x.lower() if type(x) == str else x for x in del_values]
        if 'missing' in del_values:
            data = data.loc[data[var].notnull(),]
        del_values = [x for x in del_values if x != 'missing']
        if del_values:
            data = data.loc[np.in1d(data[var], del_values, invert=True),]
        if points:
            new_categories = [x for x in list(data['bin'].cat.categories) if (data['bin'] == x).sum() > 0]
            data['bin'] = pd.Categorical(data['bin'].astype(str), categories=new_categories, ordered=True)

    if target:
        data = data.groupby('bin').agg({target: ['count', 'sum']})
        data.columns = ['Total', 'Targets', ]
        data['Population(%)'] = round(100 * data['Total'] / data['Total'].sum(), 2)
        data['Target Rate(%)'] = round(100 * data['Targets'] / data['Total'], 2)
        not_targets = (data['Total'] - data['Targets'])
        data['WoE'] = np.log((data['Targets'] / data['Targets'].sum()) / (not_targets / not_targets.sum()))
        data['Group_IV'] = 100 * ((data['Targets'] / data['Targets'].sum()) - (not_targets / not_targets.sum())) * \
                           data['WoE']
        data['Total_IV'] = data['Group_IV'].sum()
        data.reset_index(inplace=True)

        if not binned:
            data = data.sort_values('WoE', ascending=True).reset_index(drop=True)
            data['bin'] = pd.Categorical(data['bin'], categories=data['bin'], ordered=True)
    else:
        data = pd.DataFrame(data.groupby('bin').size())
        data.columns = ['Total']
        data['Population(%)'] = round(100 * data['Total'] / data['Total'].sum(), 2)
        data.reset_index(inplace=True)
        if not binned:
            data = data.sort_values('Population(%)', ascending=True).reset_index(drop=True)
            data['bin'] = pd.Categorical(data['bin'], categories=data['bin'], ordered=True)

    data.reset_index(inplace=True)
    data.columns = ['Number', 'Name', 'Total', 'Bads', 'Population(%)', 'Target Rate(%)', 'WoE', 'Group_IV',
                    'Total_IV'] if target else ['Number', 'Name', 'Total', 'Population(%)']

    if plot:
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


def barline_plot(data, x, y1, y2=None, annot1=True, annot2=False, rnd: int = 2, plot_config=None):
    """
    Plot Bar plot with or not line plot, if y2 is not None
    Parameters
    ----------
    data: pd.DataFrame, dataframe with aggregated information
    x: str, name of x variable
    y1: str, name of y1 variable
    y2: str, name of y2 variable
    annot1
    annot2
    rnd
    plot_config

    Returns
    -------
    """
    base_config = BASE_CONFIG.__copy__()
    base_config.title = 'Bar Plot'
    base_config.xlabel = x
    base_config.ylabel = y1
    base_config.y2label = y2
    base_config.colormap = 'GnBu'
    base_config.annotation_delta = 1
    plot_config = base_config if plot_config is None else base_config.merge_other_config(plot_config)

    data = data.copy().reset_index(drop=True)
    with plt.style.context(plot_config.style):
        fig = plt.figure(figsize=plot_config.figure_size)
        ax1 = fig.add_subplot(211)
        palette = None
        if plot_config.color is None:
            palette = sns.color_palette(plot_config.colormap, data.shape[0]).as_hex()
            palette = reorder_palette(palette, data[y2 or y1])
        sns.barplot(x=x, y=y1, ax=ax1, data=data, color=plot_config.color, palette=palette)

        if y2:
            ax2 = ax1.twinx()
            sns.pointplot(x=x, y=y2, data=data, ax=ax2, color=plot_config.color2)

        if annot1:
            for i in range(data.shape[0]):
                val = data.loc[i, y1]
                y_val = val + plot_config.annotation_delta
                ax1.text(i, y_val, str(round(val, rnd)), ha='center', size=plot_config.annotation_font_size)

        if annot2 and y2:
            for i in range(data.shape[0]):
                val = data.loc[i, y2]
                y_val = val + plot_config.annotation_delta
                ax2.text(i, y_val, str(round(val, rnd)), ha='center', size=plot_config.annotation_font_size)

        plot_config.appy_configs(fig, ax1, ax2)
        plt.show()
