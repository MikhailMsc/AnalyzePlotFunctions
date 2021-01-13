from binning import bin_variable
from config import BASE_CONFIG
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
import matplotlib.gridspec as grd


def cross_var_plot(data, var1, var2, target=None, binargs=dict(), annotation=True, histogram=True, colorbar=True,
                   show_min=1, rnd=0, table=False, plot_config=None):
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
        base_config.wspace = 0.05
        base_config.bar_width = 0.99
        base_config.color = 'Green'
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
        stat_frame = stat_frame.agg({target: ['size', 'sum']})
        stat_frame.columns = ['size', 'sum']
    else:
        stat_frame = stat_frame.size().to_frame('size')
        stat_frame['sum'] = 0

    stat_frame.reset_index(inplace=True)
    stat_frame['prc'] = 100 * stat_frame['size'] / stat_frame['size'].sum()
    stat_frame['rate'] = 100 * stat_frame['sum'] / stat_frame['size']

    stat_frame = stat_frame[stat_frame['prc'] >= show_min]
    tmp1 = stat_frame.pivot(var2, var1, 'rate')
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
                    rate = 'Rate: {}'.format(rnd(stat_frame.loc[indx, 'rate'].values[0]))
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
                              wspace=plot_config.wspace)

            ax_main = fig.add_subplot(gs[1, 0])
            ax_hist1 = fig.add_subplot(gs[0, 0], sharex=ax_main)
            ax_hist2 = fig.add_subplot(gs[1, 1], sharey=ax_main)
            if colorbar:
                ax_bar = fig.add_subplot(gs[2, 0])
            else:
                ax_bar = None
        else:
            ax_main = fig.add_subplot(211)

        g = sns.heatmap(tmp1 if target else tmp2, cmap=plot_config.colormap, cbar=False,
                        linewidths=plot_config.line_widths,
                        annot=labels, annot_kws={'size': plot_config.annotation_font_size}, fmt='', ax=ax_main)
        g.invert_yaxis()

        if histogram:
            stat_frame_v2 = stat_frame.groupby(var2)['size', 'sum'].sum().reset_index()

            plot_config.color = 'Green'
            ax_hist2.barh(y=np.arange(stat_frame_v2.shape[0]) + 0.5, height=plot_config.bar_width,
                          width=stat_frame_v2['size'], align='center', color=plot_config.color)
            plt.setp(ax_hist2.get_yticklabels(), visible=False)

            stat_frame_v1 = stat_frame.groupby(var1)['size', 'sum'].sum().reset_index()
            ax_hist1.bar(x=np.arange(stat_frame_v1.shape[0]) + 0.5, width=plot_config.bar_width,
                         height=stat_frame_v1['size'], align='center', color=plot_config.color)
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


# def cross_var_plot(data, v1, v2, v3=None, cfunc:'population,v3func'='population', hist=False,
# 	v3func:"max,min,median,mean,or some other returning single value" = 'mean',
# 	prc_min=1, rnd=0, table=False, params={}):
#
# 	rnd = lambda x,y=rnd: round(x,y)
# 	tmp_params = default_plot_params.copy()
# 	tmp_params['fig_size'] = (15,12) if hist else (15,20)
# 	tmp_params['title'] = 'Cross: ' + v1.title() + ' vs ' + v2.title()
# 	tmp_params['xlabel'] = v1.title()
# 	tmp_params['ylabel'] = v2.title()
# 	tmp_params['style'] = 'seaborn-darkgrid'
# 	tmp_params['yrotation'] = 30
# 	tmp_params['xrotation'] = 0
# 	tmp_params['color2'] = '#99b87a'
# 	tmp_params['title_pad'] = 0.94 if hist else 1.02
# 	tmp_params['yinverse'] = True
#
# 	params = {**tmp_params,**params}
#
# 	tmp = data.groupby([v1,v2])
# 	if v3:
# 		tmp = tmp.agg({v3:['size','sum', v3func]})
# 		tmp.columns = ['size','sum', 'v3f']
# 		sv = (v3func.__name__ if v3func.__name__ != '<lambda>' else 'v3f')  if type(v3func).__name__ == 'function' else v3func
# 	else:
# 		tmp = tmp.size().to_frame('size')
# 		tmp['sum'] = 0
#
# 	tmp.reset_index(inplace=True)
# 	tmp['prc'] = 100 * tmp['size'] / tmp['size'].sum()
# 	tmp = tmp[tmp['prc'] >= prc_min]
# 	tmp[v1] = pd.Categorical(tmp[v1].astype(str),
# 		categories=[x for x in tmp[v1].cat.categories if x in list(tmp[v1])],ordered=True)
# 	tmp[v2] = pd.Categorical(tmp[v2].astype(str),
# 		categories=[x for x in tmp[v2].cat.categories if x in list(tmp[v2])],ordered=True)
#
# 	tmp1 = tmp.pivot(v2,v1,'prc')
# 	if v3:
# 		tmp2 = tmp.pivot(v2,v1,'v3f')
# 		for x in tmp2.columns: tmp2[x] = np.where(tmp1[x].notnull() & tmp2[x].isnull(),0,tmp2[x])
#
#
#
# 	if hist:
# 		agg_dict = {**{v2:'size'},**({} if not v3 else {v3:v3func})}
# 		rename_dict = {**{v2:'size'},**({} if not v3 else {v3:'v3f'})}
# 		tmp_v1 = data.groupby(v1).agg(agg_dict)
# 		tmp_v2 = data.groupby(v2).agg(agg_dict)
# 		tmp_v1, tmp_v2 = tmp_v1.rename(columns=rename_dict).reset_index(), tmp_v2.rename(columns=rename_dict).reset_index()
# 		tmp_v1['prc'] = 100 * tmp_v1['size'] / tmp_v1['size'].sum()
# 		tmp_v2['prc'] = 100 * tmp_v2['size'] / tmp_v2['size'].sum()
# 		tmp_v1 = tmp_v1[tmp_v1[v1].isin(list(tmp[v1].unique()))].reset_index(drop=True)
# 		tmp_v2 = tmp_v2[tmp_v2[v2].isin(list(tmp[v2].unique()))].reset_index(drop=True)
# 		#return tmp_v1,tmp_v2
# 		tmp_v1[v1] = pd.Categorical(tmp_v1[v1].astype(str),
# 			categories=[x for x in tmp_v1[v1].cat.categories if x in list(tmp_v1[v1])],ordered=True)
# 		tmp_v2[v2] = pd.Categorical(tmp_v2[v2].astype(str),
# 			categories=[x for x in tmp_v2[v2].cat.categories if x in list(tmp_v2[v2])],ordered=True)
# 		#return tmp1, tmp_v1,tmp_v2
#
# 	labels = []
# 	lb_format = '\n'.join(x for x in ['Prc: {prc}','' if not v3 else sv.title() + ':{v3f}'] if x)
# 	for lb2 in list(tmp[v2].cat.categories):
# 		if (tmp[v2] == lb2).sum() == 0: continue
# 		tmp_lbs = []
# 		for lb1 in list(tmp[v1].cat.categories):
# 			if (tmp[v1] == lb1).sum() == 0: continue
# 			indx = (tmp[v1] == lb1) & (tmp[v2] == lb2)
# 			if indx.sum() > 0:
#
# 				prc = rnd(tmp.loc[indx,'prc'].values[0])
# 				v3f = '' if not v3 else rnd(tmp.loc[indx,'v3f'].values[0]) or '-'
# 				lb = lb_format.format(prc=prc,v3f=v3f)
# 				tmp_lbs.append(lb)
# 			else: tmp_lbs.append(None)
# 		labels.append(tmp_lbs)
#
# 	labels = np.array(labels)
#
# 	fig = plt.figure(figsize=params['fig_size'])
# 	if hist:
# 		gs = gridspec.GridSpec(6, 6)
# 		with plt.style.context((params['style'])) :
# 			ax2 = fig.add_subplot(gs[1:6, 0:5])
# 			for spine in ax2.spines.values(): spine.set_edgecolor('black')
# 		with plt.style.context((params['style2'])):
# 			ax1 = fig.add_subplot(gs[0, :5],sharex=ax2)
# 			ax3 = fig.add_subplot(gs[1:6, -1],sharey=ax2)
# 	else:
# 		with plt.style.context((params['style'])) : ax2 = fig.add_subplot(211)
# 	ax2 = sns.heatmap(tmp1 if not v3 or cfunc == 'population' else tmp2, cmap=params['cmap'], cbar=False, linewidths=params['line_widths'],
# 		annot=labels, annot_kws={'size': params['annot_size']}, fmt='', ax=ax2)
#
# 	if not hist:
# 		ax_divider = make_axes_locatable(ax2)
# 		cax = ax_divider.append_axes('right', size = '{}%'.format(params['cbar_width']), pad = '{}%'.format(params['cbar_pad']))
# 		colorbar(ax2.get_children()[0], cax = cax, orientation = 'vertical')
# 		cax.xaxis.set_visible(False)
# 		#ax2.set_title(params['title'], fontdict={'fontsize': params['title_size']}, y=params['title_pad'])
# 	else:
# 		#ax1.set_facecolor(params['hbg'])
# 		#ax3.set_facecolor(params['hbg'])
# 		if params['cmap2']:
# 			palette1 = sns.color_palette(params['cmap2'], tmp_v1.shape[0] + 4).as_hex()
# 			indx = {v:i for i,v in enumerate(np.argsort(tmp_v1['prc'] if not v3 or cfunc == 'population' else tmp_v1['v3f']))}
# 			palette1 = [palette1[indx[i]] for i in range(len(palette1)) if i in indx]
#
# 			palette2 = sns.color_palette(params['cmap2'], tmp_v2.shape[0] + 4).as_hex()
# 			indx = {v:i for i,v in enumerate(np.argsort(tmp_v2['prc'] if not v3 or cfunc == 'population' else tmp_v2['v3f']))}
# 			palette2 = [palette2[indx[i]] for i in range(len(palette2)) if i in indx]
#
# 		ax1.bar([x+0.5 for x in tmp_v1.index], tmp_v1['prc'], width=0.98, align='center', color= params['color2'] or palette1)
# 		ax3.barh([x + 0.5 for x in tmp_v2.index], tmp_v2['prc'], height=0.98, color=params['color2'] or palette2)
# 		for i,x in enumerate(tmp_v1['prc']):
# 			lb = '{}%'.format(rnd(x))
# 			ax1.text(i+0.5, x/2, lb, ha='center', size=params['annot_size'])
#
# 		for i,x in enumerate(tmp_v2['prc']):
# 			lb = '{}%'.format(rnd(x))
# 			ax3.text(x/2,i+0.5, lb, ha='center', size=params['annot_size'])
#
# 		fig.suptitle(params['title'], fontsize=params['title_size'], y=params['title_pad'])
# 		del params['title']
# 		plot_config(ax1,{'xtick_visible':False,'ytick_visible':False})
# 		plot_config(ax3,{'xtick_visible':False,'ytick_visible':False})
#
# 	plot_config(ax2,params)
#
# 	plt.show()
#
#
# 	if table:
# 		ret_tables = [tmp, tmp1]
# 		if v3: ret_tables.append(tmp2)
# 		if hist: ret_tables += [tmp_v1,tmp_v2]
# 		return ret_tables
#
