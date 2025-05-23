from typing import List, Dict

import pandas as pd
from sklearn.tree import DecisionTreeClassifier

from analyzer.utils.domain.columns import C_VALUE, C_PERCENT, C_TARGET, C_COUNT
from analyzer.utils.domain.const import MIN, MAX
from analyzer.utils.domain.validate import validate_binary_target, validate_column_for_binning
from analyzer.utils.framework_depends import (
    SH_ValueCounts, value_counts, get_shape, filter_missing_df,
    series_to_list, len_series, get_max, convert_df_to_pandas,
    concat_series_to_frame, get_count_missing, get_columns, get_series_from_df
)
from analyzer.utils.general.types import Series, DataFrame
from analyzer.utils.general.utils import get_accuracy, pretty_round

from .params import BinningParams, DEFAULT_BIN_PARAMS


def get_all_vars_cutoffs(
        df: DataFrame, columns: List[str] = None, target_name: str = None,
        bin_params: BinningParams = DEFAULT_BIN_PARAMS
) -> Dict[str, list]:
    """
    Функция для получения точек бинаризации по множеству переменных (колонок).
    Args:
        df:                 датафрейм, для колонок которого хотим получить точки бинаризации
        columns:            список колонок для бинаризации, если не задать, то будет анализировать все, кроме таргета
        target_name:        название колонки с таргетом, опционально
        bin_params:         параметры для бининга
    Returns:
        Словарь:
            Ключ = название переменной
            Значение = список точек бинаризации
    """
    if target_name is not None:
        validate_binary_target(get_series_from_df(df, target_name))

    if columns is None:
        columns = get_columns(df)

    if target_name is not None:
        columns = [col for col in columns if col != target_name]
        target = df[target_name]
    else:
        target = None

    vars_cutoffs = {
        col: bins for col in columns
        if (bins := get_var_cutoffs(
            get_series_from_df(df, col), target, bin_params,
            False, col, False)
            )
    }
    return vars_cutoffs


def get_var_cutoffs(
        variable: Series, target: Series = None, bin_params: BinningParams = DEFAULT_BIN_PARAMS,
        _validate_target: bool = True, _var_name: str = '',
        _raise_not_numeric: bool = True
) -> list:
    """
    Расчет точек бинаризации числовой переменной.
    Args:
        variable:           переменная для бинаризации
        target:             таргет для бинаризации, опционален
        bin_params:         параметры для бининга
        _validate_target:   проверка таргета на бинарность.
        _var_name:          название переменной, вспомогательный параметр,
                            помогает отслеживать выполнение функции на конкретной переменной (будет принтиться ее нейминг в логе)
        _raise_not_numeric: поднимать исключение если переменная не числовая, скрытый параметр

    Returns:
        list:               числа (точки бинаризации) + спец значения (MIN, MAX)
    """
    is_valid = validate_column_for_binning(variable, _var_name, _raise_not_numeric)
    if not is_valid:
        return []

    min_prc = max(bin_params.min_prc, 0.001)
    cnt_missing = get_count_missing(variable)
    cnt_total = len_series(variable)
    cnt_not_missing = cnt_total - cnt_missing

    if 100 * cnt_not_missing/cnt_total < min_prc:
        return [MIN, MAX]

    stats = value_counts(variable, sort=True)
    max_prc = get_max(stats[C_PERCENT.n])

    if max_prc >= 100 - min_prc:
        return [MIN, MAX]

    stats = filter_missing_df(stats, columns=[C_VALUE.n])
    if get_shape(stats)[0] == 1:
        return [MIN, MAX]

    if target is None:
        cutoffs = _get_var_cutoffs_no_target(stats, min_prc)
    else:
        cutoffs = _get_var_cutoffs_with_target(variable, target, min_prc, _validate_target)

    if bin_params.rnd is not None:
        cutoffs = sorted(set([pretty_round(val, bin_params.rnd) for val in cutoffs]))

    cutoffs = [MIN, ] + cutoffs + [MAX, ]
    return cutoffs


def _get_var_cutoffs_no_target(stats: SH_ValueCounts.t, min_prc: float) -> list:
    stats = zip(series_to_list(stats[C_VALUE.n]), series_to_list(stats[C_PERCENT.n]))
    cutoffs = []
    current_group = [None, 0]
    round_numbers = get_accuracy(min_prc)

    for val, prc in stats:
        if round(prc, round_numbers) < min_prc:
            current_group[1] += prc
            current_group[0] = val if current_group[0] is None else val

            if round(current_group[1], round_numbers) >= min_prc:
                cutoffs.append(current_group[0])
                current_group = [val, 0]
        else:
            if current_group[1] >= 0.8 * min_prc:
                cutoffs.append(current_group[0])
            cutoffs.append(val)
            current_group = [val, 0]

    if current_group[1] >= 0.8 * min_prc:
        cutoffs.append(current_group[0])

    _ = cutoffs.pop(-1)
    if round_numbers == 0:
        cutoffs = [int(v) for v in cutoffs]
    return cutoffs


def _get_var_cutoffs_with_target(variable: Series, target: Series, min_prc: float, validate_target: bool) -> list:
    if validate_target:
        validate_binary_target(target)

    cnt_total_origin = len_series(variable)
    min_count = round(min_prc * cnt_total_origin / 100)

    df = concat_series_to_frame([variable, target], columns=[C_VALUE.n, C_TARGET.n])
    df = filter_missing_df(df, columns=[C_VALUE.n])
    df = convert_df_to_pandas(df)
    value_min = df[C_VALUE.n].min()
    value_max = df[C_VALUE.n].max()

    tree = DecisionTreeClassifier(min_samples_leaf=min_count, random_state=777)
    tree.fit(df[[C_VALUE.n]], df[C_TARGET.n])

    cutoffs = pd.Series(tree.tree_.threshold).value_counts().to_frame(C_COUNT.n).reset_index()
    cutoffs.columns = [C_VALUE.n, C_COUNT.n]

    indx = (cutoffs[C_COUNT.n] == 1) & (cutoffs[C_VALUE.n] < value_max) & (cutoffs[C_VALUE.n] > value_min)
    cutoffs = cutoffs.loc[indx, C_VALUE.n].sort_values().to_list()
    return cutoffs
