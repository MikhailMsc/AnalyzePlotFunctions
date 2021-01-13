import numpy as np
import scipy.stats as sts
import pandas as pd
import scipy.spatial.distance as distance
import scipy.cluster.hierarchy as hierarchy
from binning import bin_variable
from config import PlotConfig, BASE_CONFIG
import seaborn as sns
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable


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

        plot_config.appy_configs(fig=fig, ax=ax, cax=cax)
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
