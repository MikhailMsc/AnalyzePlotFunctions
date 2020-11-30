import numpy as np
import pandas as pd
import scipy.stats as sts
from sklearn import tree
import scipy.cluster.hierarchy as hierarchy
import scipy.spatial.distance as distance
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
import seaborn as sns
from config import BASE_CONFIG, PlotConfig
import matplotlib.gridspec as grd


def cramer_stat(var1: pd.Series, var2: pd.Series, method: "old, new" = 'old') -> np.float64:
    """
    Calculate Cramers V statistic for categorial-categorial association.
    There 2 methods: old - usual, new - correction from Bergsma, Wicher

    Keyword arguments:
    var1 (pd.Series) -- First categorical variable
    var2 (pd.Series) -- Second categorical variable
    method (str: old/new) -- type of algorithm calculation
    """
    confusion_matrix = pd.crosstab(var1, var2)
    chi2 = sts.chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    phi2 = chi2 / n
    r, k = confusion_matrix.shape
    if method == 'old':
        return np.sqrt(phi2 / (min(r, k) - 1))
    else:
        phi2correct = max(0, phi2 - ((k - 1) * (r - 1) / (n - 1)))
        rcorrect = r - ((r - 1) ** 2) / (n - 1)
        kcorrect = k - ((k - 1) ** 2) / (n - 1)
        return np.sqrt(phi2correct / (min(rcorrect, kcorrect) - 1))


def points_calulation(var: pd.Series, min_size: float = 5, rnd: int = 2, not_join_threshold: float = None) -> list:
    """
    Calculate points for binning numeric variable without target variable.

    Keyword arguments:
        var (pd.Series) -- Numeric variable
        min_size (float) -- minimum size of group in percent
        rnd -- Round level for variable values (default 2)
        not_join_threshold (float) - Minimum size of group, that don't join to bigger group
    """
    var_name = var.name
    if var.isnull().sum() == var.shape[0]:
        print('WARNING! Variable "{vname}" does not have valid values!'.format(vname=var_name))
        return [-np.inf, np.inf]

    if not_join_threshold is None:
        not_join_threshold = min_size

    size = var.shape[0]
    var = var.value_counts(dropna=True).sort_index(ascending=True).reset_index()
    var.columns = ['value', 'cnt']
    var['value'] = var['value'].round(rnd)
    var = var.groupby('value')['cnt'].sum().reset_index()
    var['prc'] = 100 * var['cnt'] / size
    var['cumm_prc'] = var['prc'].cumsum()
    var['group'] = var['cumm_prc'] // min_size
    var['group'] = var[['prc', 'group']].apply(lambda x: x[1] - 0.5 if x[0] > min_size else x[1], axis=1)
    var = var.groupby('group').agg({'prc': 'sum', 'value': 'max'}).sort_index(ascending=True).reset_index()

    for i in range(var.shape[0]):
        if round(var['prc'][i]) < not_join_threshold and var.shape[0] > 1:
            if i == 0:
                var['group'][0] = var['group'][1]
            elif i == (var.shape[0] - 1):
                var['group'][i] = var['group'][i - 1]
            else:
                var['group'][i] = var['group'][i - 1] if var['prc'][i - 1] < var['prc'][i + 1] else var['group'][i + 1]

    var = var.groupby('group').agg({'prc': 'sum', 'value': 'max'}).sort_index(ascending=True).reset_index()
    if var.shape[0] == 1:
        print('WARNING! Only possible to bin "{vname}" as [-inf, inf]!'.format(vname=var_name))
        cut_points = [-np.inf, np.inf]
    else:
        cut_points = [-np.inf, ] + list(var['value'])[:-1] + [np.inf, ]
    return cut_points


def points_calculation_tree(var: pd.Series, target: pd.Series, min_size: float = 5, rnd: int = 2) -> list:
    """
    Calculate points for binning numeric variable dependent on target variable.

    Keyword arguments:
        var (pd.Series) -- Numeric variable
        target (pd.Series) -- Target binary variable
        min_size (float) -- minimum size of group in percent
        rnd -- Round level for variable values (default 2)
    """
    size = var.shape[0]
    var_name = var.name
    if (var.isnull().sum() / size) > (1 - min_size/100):
        print('WARNING! Variable "{vname}" has too much null values!'.format(vname=var_name))
        return [-np.inf, np.inf]
    elif (var.value_counts(dropna=True).max() / size) > (1 - min_size/100):
        print('WARNING! Variable "{vname}" has too often one value!'.format(vname=var_name))
        return [-np.inf, np.inf]
    else:
        indx = var.notnull()
        var = var[indx]
        target = target[indx]
        xmin = var.min()
        xmax = var.max()
        min_samples = round(min_size * size / 100)
        clf = tree.DecisionTreeClassifier(min_samples_leaf=min_samples, random_state=777)
        clf.fit(var.to_frame(), target)
        cut_points = pd.Series(clf.tree_.threshold).value_counts().to_frame('cnt').reset_index()
        cut_points.columns = ['point', 'cnt']
        cut_points = cut_points.loc[cut_points['cnt'] == 1, :]
        cut_points = cut_points.loc[(cut_points['point'] <= xmax) & (cut_points['point'] >= xmin), :]
        cut_points = cut_points['point'].sort_values(ascending=True).values
        cut_points = [-np.inf, ] + list(cut_points.round(rnd)) + [np.inf, ]
        return cut_points


def get_thresholds_for_data(data: pd.DataFrame, target: str = None, min_size: float = 5, rnd: int = 2) -> dict:
    """
    Get all cutpoints on the frame.

    Keyword arguments:
        data (pd.DataFrame) -- input data frame
        target (str) -- target column name
        min_size (float) -- minimum size of group in percent
        rnd -- Round level for variable values (default 2)

    Output:
        dict - {"varname1": [cutpoint1, cutpoint2, ...], "varname2": [cutpoint1, cutpoint2, ...], ...}
    """
    columns = [x for x in data.columns if x != target]
    if target is None:
        cut_f = lambda x: points_calulation(var=data[x], min_size=min_size, rnd=rnd)
    else:
        cut_f = lambda x: points_calculation_tree(var=data[x], target=data[target], min_size=min_size, rnd=rnd)

    return {col: cut_f(col) for col in columns}


def bin_variable(var: pd.Series, target: pd.Series = None, points: list = None, min_size: float = 5,
                 rnd: int = 2, return_points: bool = False):
    """
    Make binning on numeric variable, return categorized variable

    Keyword arguments:
        var (pd.Series) -- Numeric variable
        target (pd.Series) -- Target binary variable
        points (list) -- List of cut-points (default None)
        min_size (float) -- minimum size of group in percent
        rnd -- Round level for variable values (default 2)

    Output:
        List  - [cut-points (list), categorized variable (pd.Series)]
            or
        Categorized variable (pd.Series)
    """
    if points is None:
        if target is None:
            points = points_calulation(var=var, min_size=min_size, rnd=rnd)
        else:
            points = points_calculation_tree(var=var, target=target, min_size=min_size, rnd=rnd)

    if points == [-np.inf, np.inf]:
        var = pd.cut(var, bins=points, labels=['Not Missing'])
    else:
        points = list([-np.inf, ] if points[0] != -np.inf else []) + points + list(
            [np.inf, ] if points[-1] != np.inf else [])
        var = pd.cut(var, bins=points, include_lowest=False)

    categories = [str(x) for x in var.cat.categories]
    if var.isnull().sum() > 0:
        var = var.cat.add_categories('Missing')
        var.fillna('Missing', inplace=True)

    var = var.astype(str)
    if len(categories) > 1:
        var[var == categories[0]] = '[<={}]'.format(points[1])
        var[var == categories[-1]] = '(>{}]'.format(points[-2])

        categories[0] = '[<={}]'.format(points[1])
        categories[-1] = '(>{}]'.format(points[-2])

    if (var == 'Missing').sum() > 0:
        categories = ['Missing', ] + categories
    var = pd.Categorical(var, categories=categories, ordered=True)
    return (points, var) if return_points else var


def correlation(data: pd.DataFrame, select_vars: list = None, ignore_vars: list = None,
                method: "spearman, pearson, kendal, cramer_old, cramer_new" = 'spearman',
                already_bin: bool = False, binargs: dict = {}) -> pd.DataFrame:
    """
    Calculate correlation matrix from data frame
    Metods:
        spearman, pearson, kendal - only for numeric variables. It will filter all non-numeric variables
        cramer_old, cramer_new - for categorical variables

    Keyword arguments:
        data (pd.DataFrame) -- input data frame
        select_vars (list) -- List of variables, that you want to calculate correlation
        ignore_vars (list) -- List of variables, that you don't want to calculate correlation
        method (str) -- Method to calculate correlation
        already_bin (bool) -- Flag, is variables already categorized
        binargs (dict) -- parameters for binning functions
    """
    if select_vars is None:
        select_vars = list(data.columns)
    if ignore_vars is None:
        ignore_vars = []

    select_vars = [col for col in select_vars if col not in ignore_vars]

    if method in ['spearman', 'pearson', 'kendal']:
        corrs = data[select_vars].corr(method=method)
    elif method in ['cramer_old', 'cramer_new']:
        if not already_bin:
            print('Start categorizing variables.')
            dtypes = data[select_vars].dtypes
            need_transform = [var for var in select_vars if 'float' in str(dtypes[var]) or 'int' in str(dtypes[var])]
            if need_transform:
                if 'target' in binargs:
                    binargs['target'] = data[binargs['target']]
                binning = lambda x: bin_variable(x, **binargs)
                data = data.copy()
                data.loc[:, need_transform] = data.loc[:, need_transform].apply(binning, 0)
            data = data[select_vars]
        print('Start calculate correlations.')
        corrs = pd.DataFrame(index=select_vars, columns=select_vars)

        def calc_corr(x):
            vars, start_var = list(x.index), x.name
            results = []
            for i, var in enumerate(vars):
                if i > vars.index(start_var):
                    results.append(cramer_stat(data[start_var], data[var], method=method.split('_')[1]))
                else:
                    results.append(np.nan)
            return pd.Series(results, index=vars)

        corrs = corrs.apply(calc_corr, axis=1)
        i_lower = np.tril_indices(len(select_vars), -1)
        corrs.values[i_lower] = corrs.values.T[i_lower]
        corrs.fillna(1, inplace=True)
    else:
        raise Exception('Wrong correlation method!')
    return corrs


def order_corr(matrix:pd.DataFrame, method='complete', metric='euclidean'):
    """
    Function to reorder correlation matrix, based on hierarchy clusterization.
    :param matrix: Correlation matrix
    :param method: single, complete, average, weighted, centroid, median, ward...
    :param metric: euclidean, minkowski, cityblock, seuclidean, sqeuclidean, cosine, correlation, hamming...
    :return: list of oredered positions of variables
    """
    d = distance.pdist(matrix)
    d = distance.squareform(d)
    y = hierarchy.linkage(d, method=method, metric=metric)
    z = hierarchy.dendrogram(y, no_plot=True)
    return z['leaves']


def plot_corr(matrix: pd.DataFrame, annotation: bool = False, reorder: bool = False,
              method='single', metric='euclidean', triangle: bool = None, rnd: int = 1,
              plot_config: PlotConfig = None):
    """
    Plot correlation matrix.
    methods : single, complete, average, weighted, centroid, median, ward...
    metrics: euclidean, minkowski, cityblock, seuclidean, sqeuclidean, cosine, correlation, hamming...

    :param matrix: Correlation matrix
    :param annotation: Show annotation on the plot
    :param reorder: Reorder variablese based on the clusterization
    :param method: Method of clusterization
    :param metric: Metric for clusterization
    :param triangle: Show only upper/lower triangle
    :param rnd: Round level for annotation values (default 1)
    :param plot_config: Configuration for plot
    :return: None
    """
    base_config = BASE_CONFIG.__copy__()
    if matrix.min().min() >= 0:
        base_config.colormap = 'Blues'
    else:
        base_config.colormap = 'coolwarm'
    plot_config = base_config if plot_config is None else base_config.merge_other_config(plot_config)
    matrix = matrix.astype(float)

    if reorder:
        order = order_corr(matrix, method, metric)
        matrix = matrix.iloc[order, order]

    if triangle:
        mask = np.zeros_like(matrix, dtype=np.bool)
        indx = lambda x: np.triu_indices_from(x) if triangle == 'lower' else np.tril_indices_from(x)
        mask[indx(mask)] = True
        if triangle == 'lower':
            mask = mask[1:, :(mask.shape[0] - 1)]
            matrix = matrix.iloc[1:, :(matrix.shape[0] - 1)]
        else:
            mask = mask[:(mask.shape[0] - 1), 1:]
            matrix = matrix.iloc[:(matrix.shape[0] - 1), 1:]

    with plt.style.context(plot_config.style):
        fig = plt.figure(figsize=plot_config.figure_size)
        ax = fig.add_subplot(211)
        sns.heatmap(
            matrix,
            mask=mask if triangle else None,
            cmap=plot_config.colormap,
            cbar=False, square=True,
            linewidths=plot_config.line_widths,
            annot=annotation,
            annot_kws={'size': plot_config.annotation_font_size}, fmt='.{}f'.format(rnd),
            ax=ax)

        if plot_config.cbar_location in ['right', 'left']:
            cbar_orientation = 'vertical'
        else:
            cbar_orientation = 'horizontal'

        ax_divider = make_axes_locatable(ax)
        cax = ax_divider.append_axes(
            plot_config.cbar_location,
            size='{}%'.format(plot_config.cbar_width),
            pad='{}%'.format(plot_config.cbar_pad)
        )
        plt.colorbar(ax.get_children()[0], cax=cax, orientation=cbar_orientation)
        if cbar_orientation == 'horizontal':
            cax.yaxis.set_visible(False)
        else:
            cax.xaxis.set_visible(False)

        plot_config.appy_configs(ax, cax=cax)
        plt.show()


def analyze_corr(matrix: pd.DataFrame, top: int = 10) -> pd.DataFrame:
    """
    Analyze correlation matrix. Return top correlated variables for each variable.
    :param matrix: Correlation matrix
    :param top: Select top correlated variables
    :return: pd.DataFrame
    """
    f = lambda x: get_top(x, top=top)
    return matrix.apply(f, 1).reset_index(drop=True)


def get_top(x, top):
    x = x[x.index != x.name]
    name = x.name
    max_abs = x.abs().max()
    mean = x.abs().mean()
    median = x.abs().median()
    indx = x.abs().sort_values(ascending=False).index
    x = pd.DataFrame(x.round(2).reindex(indx)).reset_index()
    x.columns = ['col1', 'col2']
    x['col3'] = x['col1'] + '(' + x['col2'].map(str) + ')'
    top = x.shape[0] if top > x.shape[0] else top
    x = x['col3'][0:top]
    x.index = ['top_' + str(i+1) for i in range(top)]
    x = pd.Series([name, max_abs, mean, median], index=['Variable', 'ABS_MAX', 'MEAN', 'MEDIAN']).append(x)
    x.name = name
    return x


def var_stat(data: pd.DataFrame, var: str, target: str = None, binargs: dict = dict(), del_values: list = None,
             plot=True, xticks='name', annotation: bool = True, plot_config=None,
             print_points=True, return_table=True):
    """

    :param data:
    :param var:
    :param y:
    :param annot:
    :param binargs:
    :param xticks:
    :param rnd:
    :param del_values:
    :param plot:
    :param return_points:
    :param return_table:
    :param params:
    :return:
        pd.DataFrame - variable stats in table
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
            **{'target': data[target] if target else None, 'return_points': True},
            **binargs}
        points, data['bin'] = bin_variable(data[var], **binargs)
        binned = True

    if del_values:
        del_values = [x.lower() if type(x) == str else x for x in del_values]
        if 'missing' in del_values:
            data = data.loc[data[var].notnull(), ]
        del_values = [x for x in del_values if x != 'missing']
        if del_values:
            data = data.loc[np.in1d(data[var], del_values, invert=True), ]
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
                    'Total_IV'] if target else ['Name', 'Total', 'Population(%)']

    if plot:
        with plt.style.context(plot_config.style):
            fig = plt.figure(figsize=plot_config.figure_size)
            ax1 = fig.add_subplot(211)
            xticks = xticks.title()
            palette = None
            if plot_config.color is None:
                palette = sns.color_palette(plot_config.colormap, data.shape[0]).as_hex()
                palette = reorder_palette(palette, data['Target Rate(%)' if target else 'Population(%)'])
            g = sns.barplot(x=xticks, y='Population(%)', ax=ax1, data=data, color=plot_config.color,
                               palette=palette)
            #plot_config.appy_configs(ax=ax1)

            if target:
                ax2 = ax1.twinx()
                g2 = sns.pointplot(x=xticks, y='Target Rate(%)', data=data, ax=ax2, color=plot_config.color2)
                #plot_config.appy_configs(ax2=ax2)
                #plt.close(g2.fig)

            if annotation:
                for i in range(data.shape[0]):
                    x = data.loc[i, 'Number']
                    pop = round(data.loc[i, 'Population(%)'], 1)
                    rate = round(data.loc[i, 'Target Rate(%)'] if target else 0, 1)
                    lb = f'Rate={rate}%\nPop-n={pop}%' if target else 'Pop-n={pop}%'
                    y = pop + plot_config.annotation_delta
                    ax1.text(x, y, lb, ha='center', size=plot_config.annotation_font_size)

            plot_config.appy_configs(ax1, ax2)
            plt.show()

            if print_points and points:
                points = '[' + ', '.join([str(x) for x in points]) + ']'
                points = points.replace('inf', 'np.inf')
                print('\n\n', points, '\n\n')
            if return_table:
                del data['Number']
                return data


def reorder_palette(palette, order_by):
    indx = np.argsort(order_by)
    indx = {ind: i for i, ind in enumerate(indx)}
    palette = [palette[indx[i]] for i in range(len(palette))]
    return palette


def cross_var_plot(data, var1, var2, target=None, binargs=dict(), annotation=True, histogram=True,
                   show_min=1, rnd=0, table=False, plot_config=None):
    rnd = lambda x, y=rnd: round(x, y)
    base_config = BASE_CONFIG.__copy__()
    base_config.title = 'Cross: ' + var1.title() + ' vs ' + var2.title()
    base_config.xlabel = var1.title()
    base_config.ylabel = var2.title()
    base_config.y_rotation = 30
    base_config.colormap = 'Blues'
    if histogram:
        base_config.cbar_location = 'bottom'
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
        ax = fig.add_subplot(211)
        ax = sns.heatmap(tmp1 if target else tmp2, cmap=plot_config.colormap, cbar=False,
                         linewidths=plot_config.line_widths,
                         annot=labels, annot_kws={'size': plot_config.annotation_font_size}, fmt='', ax=ax)
        ax.invert_yaxis()

        if plot_config.cbar_location in ['right', 'left']:
            cbar_orientation = 'vertical'
        else:
            cbar_orientation = 'horizontal'
        ax_divider = make_axes_locatable(ax)
        cax = ax_divider.append_axes(
            plot_config.cbar_location,
            size='{}%'.format(plot_config.cbar_width),
            pad='{}%'.format(plot_config.cbar_pad)
        )
        plt.colorbar(ax.get_children()[0], cax=cax, orientation=cbar_orientation)

        if histogram:
            stat_frame_v2 = stat_frame.groupby(var2)['size', 'sum'].sum().reset_index()

            palette = None
            plot_config.color = 'Green'
            ax2 = ax_divider.append_axes('right', size='15%', pad='2%', sharey=ax)
            #g = sns.barplot(x='size', y=var2, ax=ax2, data=stat_frame_v2, color=plot_config.color,
            #                palette=palette)
            ax2.barh(np.arange(stat_frame_v2.shape[0]), stat_frame_v2['size'], align='center')


        #plot_config.appy_configs(ax=ax, cax=cax)
        plt.show()

    if table:
        return stat_frame





def cross_var_plot2(data, var1, var2, target=None, binargs=dict(), annotation=True, histogram=True, colorbar=True,
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
