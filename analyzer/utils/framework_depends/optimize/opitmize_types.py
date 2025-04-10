from analyzer.utils.general.types import DataFrame, get_framework_from_dataframe, FrameWork

from ..convert import convert_df_to_polars
from ..columns import (
    get_columns, get_series_from_df, is_numeric_column,
    is_integer_column, is_convertable_to_int_column
)
from ..max_min import get_max, get_min
from .encode import MinMaxVarsValues


def optimize_df_int_types(
        df: DataFrame, vars_min_max: MinMaxVarsValues = None, convert_to_polars: bool = False
) -> DataFrame:
    vars_min_max = vars_min_max or dict()

    framework = get_framework_from_dataframe(df)
    if framework is FrameWork.pandas and convert_to_polars:
        df = convert_df_to_polars(df)
        framework = FrameWork.polars

    for col in get_columns(df):
        if col not in vars_min_max:
            ser = get_series_from_df(df, col)
            if is_numeric_column(ser):
                if is_integer_column(ser):
                    vars_min_max[col] = (get_min(ser), get_max(ser))

                elif is_convertable_to_int_column(ser):
                    vars_min_max[col] = (get_min(ser), get_max(ser))

    func = _MAP_FRAMEWORK[framework]
    return func(df, vars_min_max)


def _is_optimize_df_int_types_pandas(df: DataFrame, vars_min_max: MinMaxVarsValues) -> DataFrame:
    import numpy as np
    int_types = {
        tp: (tp_info.min, tp_info.max)
        for tp in [np.uint8, np.int8, np.uint16, np.int16, np.uint32, np.int32, np.uint64, np.int64]
        if (tp_info := np.iinfo(tp))
    }
    vars_types = {
        col: _get_optimal_type(int_types, min_val, max_val)
        for col, (min_val, max_val) in vars_min_max.items()
    }
    df = df.astype(vars_types)
    return df


def _is_optimize_df_int_types_polars(df: DataFrame, vars_min_max: MinMaxVarsValues) -> DataFrame:
    import polars as pl
    int_types = {
        tp: (int(str(tp.min())), int(str(tp.max())))
        for tp in [pl.UInt8, pl.Int8, pl.UInt16, pl.Int16, pl.UInt32, pl.Int32, pl.UInt64, pl.Int64]
    }
    vars_types = {
        col: _get_optimal_type(int_types, min_val, max_val)
        for col, (min_val, max_val) in vars_min_max.items()
    }
    df = df.cast(vars_types)
    return df


def _is_optimize_df_int_types_spark(df: DataFrame, vars_min_max: MinMaxVarsValues) -> DataFrame:
    raise NotImplementedError


_MAP_FRAMEWORK = {
    FrameWork.pandas: _is_optimize_df_int_types_pandas,
    FrameWork.polars: _is_optimize_df_int_types_polars,
    FrameWork.spark: _is_optimize_df_int_types_spark,
}


def _get_optimal_type(dict_types, min_val, max_val):
    for tp, (min_tp, max_tp) in dict_types.items():
        if min_val < min_tp:
            continue
        elif max_val > max_tp:
            continue
        else:
            return tp
    raise Exception(f'Не смогли подобрать подходящий тип Int, вышли за все границы. {min_val = }, {max_val = }.')
