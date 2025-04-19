from typing import List, Union, Dict, TypeVar

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
        binning: BinningParamsMultiVars = True,
        map_values: MapDictMultiVars = None,
        validate_target: bool = True,
        drop_not_processed: bool = False,
        _copy: bool = True,
        _bin_by_target: bool = True
) -> DataFrame:
    if not process_vars:
        if not ignore_vars:
            ignore_vars = []

        if target_name:
            ignore_vars.append(target_name)

        process_vars = [col for col in get_columns(df) if col not in ignore_vars]

    if validate_target and target_name:
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

    for col in process_vars:
        if col in binning and not is_numeric_column(get_series_from_df(df, col)):
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

    for var_name in binning:
        variable_series = get_series_from_df(df, var_name)
        if binning[var_name] is True:
            ser = binarize_series(
                variable=variable_series,
                target=target_series,
                validate_target=False,
                _var_name=var_name
            )
        else:
            ser = binarize_series(
                variable=variable_series,
                target=target_series,
                validate_target=False,
                bin_params=binning[var_name],
                _var_name=var_name
            )
        df = set_column(df, ser, var_name)

    return df
