from typing import List, Union, Dict, TypeVar

from tqdm import tqdm

from analyzer.utils.domain.const import MISSING
from analyzer.utils.domain.validate import validate_binary_target
from analyzer.utils.framework_depends import (
    get_series_from_df, fill_missing_df, map_elements_df, set_column,
    get_columns, get_sub_df, copy_df
)
from analyzer.utils.framework_depends.columns.is_numeric_column import is_numeric_column
from analyzer.utils.general.types import DataFrame

from .binning import BinningParams, binarize_series

Variable = TypeVar('Variable')
OldValue = TypeVar('OldValue')
NewValue = TypeVar('NewValue')

MapDictSingleVar = Dict[OldValue, NewValue]
MapDictMultiVars = Dict[Variable, MapDictSingleVar]
BinningParamsSingleVars = Union[BinningParams, bool]
BinningParamsMultiVars = Union[Dict[str, Union[BinningParams, bool]], bool, BinningParams, List[str]]


def preprocess_df(
        df: DataFrame, process_vars: List[str] = None, ignore_vars: List[str] = None,
        target_name: str = None,
        map_values: MapDictMultiVars = None,
        binning: BinningParamsMultiVars = True,
        drop_not_processed: bool = False,
        _validate_target: bool = True,
        _copy: bool = True,
        _bin_by_target: bool = True,
        _tqdm: bool = True
) -> DataFrame:
    """
    Функция препроцессинга датафрейма - замена пустых значений, бинаризация, мэппинг значений.

    Args:
        df:                 Датафрейм для препроцессинга
        process_vars:       Список переменных для препроцессинга, если ничего не задать, то будут все переменные,
                            кроме таргета и игнорируемых переменных
        ignore_vars:        Список игнорируемых переменных во время препроцессинга, опционально
        target_name:        Имя таргета, опционально
        binning:            Параметры для биннинга, ниже будет более подробное описание.
        map_values:         Словарь для замены значений переменных (словарь, ключ = название переменной,
                            значение = словарь старое-новое значение)
        drop_not_processed: Дропнуть колонки, которые не участвуют в биннинге.
        _validate_target:   Проверка таргета на бинарность. Скрытый параметр.
        _copy:              Скрытый параметр. Делать биннинг на копии датафрейма (гарантия что исходный
                            датафрейм не будет изменен)
        _bin_by_target:     Скрытый параметр. Бинаризовать на основе таргета, если он задан.
        _tqdm:              Скрытый параметр. Отображать прогрессбар

    Вариации binning:
        - binning = True, будет применен применен дефолтный биннинг
        - binning = False, не применять биннинг
        - binning = BinningParams(min_prc=20), применить биннинг с такими характеристиками для всех переменных
        - binning = ['Var_1', 'Var_2', ...], будет применен применен дефолтный биннинг для переменных из списка.
        - binning = {'Var_1': True, 'Var_2': BinningParams(min_prc=20), ...} смесь кастомного и дефолтного
                    биннинга для разных переменных.

    Последовательность операций:
        1. Мэппинг старых значений на новый, если задан словарь мэппинга
        2. Биннинг, если необходим
        3. Замена пустых значений на спец. значение MISSING

    Returns:
        DataFrame
    """
    if not process_vars:
        if not ignore_vars:
            ignore_vars = []

        if target_name:
            ignore_vars.append(target_name)

        process_vars = [col for col in get_columns(df) if col not in ignore_vars]

    if _validate_target and target_name:
        validate_binary_target(get_series_from_df(df, target_name))

    if drop_not_processed:
        selected_cols = process_vars[:]
        if target_name:
            selected_cols.append(target_name)
        df = get_sub_df(df, columns=selected_cols)

    elif _copy:
        df = copy_df(df)

    if binning is True:
        binning = {col: True for col in process_vars}
    elif binning is False:
        binning = {}
    elif type(binning) is BinningParams:
        binning = {col: binning for col in process_vars}
    elif type(binning) is list:
        binning = {col: True for col in binning}

    for col in list(binning.keys()):
        if not is_numeric_column(get_series_from_df(df, col)):
            del binning[col]

    missing_dict = {var_name: MISSING for var_name in process_vars if var_name not in binning}
    if missing_dict:
        df = fill_missing_df(df, columns_values=missing_dict)

    if map_values:
        df = map_elements_df(df, map_values)

    if target_name and _bin_by_target:
        target_series = get_series_from_df(df, target_name)
    else:
        target_series = None

    if _tqdm and len(binning) != 0:
        bar_format = (f"{{l_bar}}{{bar}}| Бинаризация переменных, {{n_fmt}}/{len(binning)} "
                      f"[{{elapsed}}<{{remaining}}, {{rate_fmt}}]")
        iter_obj = tqdm(binning, total=len(binning), bar_format=bar_format)
    else:
        iter_obj = binning

    for var_name in iter_obj:
        variable_series = get_series_from_df(df, var_name)
        if binning[var_name] is True:
            ser = binarize_series(
                variable=variable_series,
                target=target_series,
                _validate_target=False,
                _var_name=var_name
            )
        else:
            ser = binarize_series(
                variable=variable_series,
                target=target_series,
                _validate_target=False,
                bin_params=binning[var_name],
                _var_name=var_name
            )
        df = set_column(df, ser, var_name)

    return df
