from binning import bin_variable
from config import BASE_CONFIG
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
import matplotlib.gridspec as grd

from utils.framework_depends import split_palette


def cross_var_plot(data, var1, var2, target=None, aggfunc=lambda x: np.mean(x) * 100,
                   binargs=dict(), annotation=True, histogram=True, colorbar=True,
                   show_min=1, rnd=0, table=False, plot_config=None):
    """

    Parameters
    ----------
    aggfunc
    data
    var1
    var2
    target
    binargs
    annotation
    histogram
    colorbar
    show_min
    rnd
    table
    plot_config

    Returns
    -------

    """
    rnd = lambda x, y=rnd: round(x, y)
    base_config = BASE_CONFIG.__copy__()
    base_config.title = 'Cross: ' + var1.title() + ' vs ' + var2.title()
    base_config.xlabel = var1.title()
    base_config.ylabel = var2.title()
    base_config.y_rotation = 30
    base_config.colormap = 'Blues'

    if histogram:
        base_config.width_ratios = [6, 1]
        base_config.height_ratios = [1, 6, 0.5] if colorbar else [1, 6]
        base_config.cbar_location = 'bottom'
        base_config.space = 0.05
        base_config.bar_width = 0.99
        base_config.side_grid = True
    else:
        base_config.cbar_location = 'right'
    plot_config = base_config if plot_config is None else base_config.merge_other_config(plot_config)

    if target:
        data = data[[var1, var2, target]]
    else:
        data = data[[var1, var2]]

    for var in [var1, var2]:
        if str(data[var].dtype) not in ['category', 'object']:
            binargs = {
                **{'target': data[target] if target else None, 'return_points': False},
                **binargs}
            data[var] = bin_variable(data[var], **binargs)
        elif str(data[var].dtype) == 'object':
            data[var].fillna('Missing', inplace=True)
            data[var] = data[var].astype('category')

    stat_frame = data.groupby([var1, var2])
    if target:
        stat_frame = stat_frame.agg({target: ['size', aggfunc]})
        stat_frame.columns = ['size', 'target']
    else:
        stat_frame = stat_frame.size().to_frame('size')

    stat_frame.reset_index(inplace=True)
    stat_frame['prc'] = 100 * stat_frame['size'] / stat_frame['size'].sum()

    stat_frame = stat_frame[stat_frame['prc'] >= show_min]
    tmp1 = stat_frame.pivot(var2, var1, 'target')
    tmp2 = stat_frame.pivot(var2, var1, 'prc')

    labels = False
    if annotation:
        labels = []
        for lb2 in list(stat_frame[var2].cat.categories):
            if (stat_frame[var2] == lb2).sum() == 0:
                continue
            tmp_lbs = []
            for lb1 in list(stat_frame[var1].cat.categories):
                if (stat_frame[var1] == lb1).sum() == 0:
                    continue
                indx = (stat_frame[var1] == lb1) & (stat_frame[var2] == lb2)
                if indx.sum() > 0:
                    prc = 'Pop-n: {}'.format(rnd(stat_frame.loc[indx, 'prc'].values[0]))
                    rate = 'Target: {}'.format(rnd(stat_frame.loc[indx, 'target'].values[0]))
                    lb = prc + '\n' + rate if target else prc
                    tmp_lbs.append(lb)
                else:
                    tmp_lbs.append(None)
            labels.append(tmp_lbs)

        labels = np.array(labels)

    with plt.style.context(plot_config.style):
        fig = plt.figure(figsize=plot_config.figure_size)
        if histogram:
            rows = 3 if colorbar else 2
            gs = grd.GridSpec(rows, 2, width_ratios=plot_config.width_ratios, height_ratios=plot_config.height_ratios,
                              wspace=plot_config.space, hspace=plot_config.space)

            ax_main = fig.add_subplot(gs[1, 0])
            ax_hist1 = fig.add_subplot(gs[0, 0], sharex=ax_main)
            ax_hist2 = fig.add_subplot(gs[1, 1], sharey=ax_main)
            if colorbar:
                ax_bar = fig.add_subplot(gs[2, 0])
            else:
                ax_bar = None
        else:
            ax_main = fig.add_subplot(211)

        cnt_colors = (tmp1 if target else tmp2).shape
        cnt_colors = cnt_colors[0] * cnt_colors[1] + ((cnt_colors[1] + cnt_colors[0]) if histogram else 0)
        palette = sns.color_palette(plot_config.colormap, cnt_colors).as_hex()
        if histogram:
            stat_frame_v1 = data.groupby(var1)
            stat_frame_v2 = data.groupby(var2)
            if target:
                stat_frame_v1 = stat_frame_v1.agg({target: ['size', aggfunc]})
                stat_frame_v2 = stat_frame_v2.agg({target: ['size', aggfunc]})
                stat_frame_v1.columns = ['size', 'target']
                stat_frame_v2.columns = ['size', 'target']
            else:
                stat_frame_v1 = stat_frame_v1.size().to_frame('size')
                stat_frame_v2 = stat_frame_v2.size().to_frame('size')

            stat_frame_v1.reset_index(inplace=True)
            stat_frame_v1['prc'] = 100 * stat_frame_v1['size'] / stat_frame_v1['size'].sum()
            stat_frame_v2.reset_index(inplace=True)
            stat_frame_v2['prc'] = 100 * stat_frame_v2['size'] / stat_frame_v2['size'].sum()
            stat_frame_v1 = stat_frame_v1[stat_frame_v1[var1].isin(list(stat_frame[var1].unique()))].reset_index(
                drop=True)
            stat_frame_v2 = stat_frame_v2[stat_frame_v2[var2].isin(list(stat_frame[var2].unique()))].reset_index(
                drop=True)
            column = 'target' if target else 'prc'

            order_values = stat_frame_v1[column].to_list() + list((tmp1 if target else tmp2).values.flatten()) + \
                           stat_frame_v2[column].to_list()
            palette_1, palette_2, palette = split_palette(palette, order_values, cnt_1=stat_frame_v1.shape[0],
                                                          cnt_2=stat_frame_v2.shape[0])
        g = sns.heatmap(tmp1 if target else tmp2,
                        cmap=palette,
                        cbar=False, linewidths=plot_config.line_widths,
                        annot=labels, annot_kws={'size': plot_config.annotation_font_size}, fmt='', ax=ax_main)
        g.invert_yaxis()

        if histogram:
            ax_hist2.barh(y=np.arange(stat_frame_v2.shape[0]) + 0.5, height=plot_config.bar_width,
                          width=stat_frame_v2['size'], align='center', color=palette_2)
            plt.setp(ax_hist2.get_yticklabels(), visible=False)

            ax_hist1.bar(x=np.arange(stat_frame_v1.shape[0]) + 0.5, width=plot_config.bar_width,
                         height=stat_frame_v1['size'], align='center', color=palette_1)
            plt.setp(ax_hist1.get_xticklabels(), visible=False)

        if colorbar:
            if plot_config.cbar_location in ['right', 'left']:
                cbar_orientation = 'vertical'
            else:
                cbar_orientation = 'horizontal'
            if not histogram:
                ax_divider = make_axes_locatable(ax_main)
                ax_bar = ax_divider.append_axes(
                    plot_config.cbar_location,
                    size='{}%'.format(plot_config.cbar_width),
                    pad='{}%'.format(plot_config.cbar_pad)
                )
            plt.colorbar(ax_main.get_children()[0], cax=ax_bar, orientation=cbar_orientation)
        else:
            ax_bar = None

        plot_config.appy_configs(fig=fig, ax=ax_main, cax=ax_bar, side_axis=[ax_hist1, ax_hist2])
        plt.show()

    if table:
        return stat_frame