import pandas as pd
import numpy as np
from sklearn import tree


def points_calulation(var: pd.Series, min_size: float = 5, rnd: int = 2, not_join_threshold: float = None) -> list:
    """
    Calculate points for binning numeric variable without target_name variable.

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
    Calculate points for binning numeric variable dependent on target_name variable.

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
        target (str) -- target_name column name
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