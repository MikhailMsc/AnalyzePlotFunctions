from typing import TypeVar, Dict

from utils.general.types import Series, get_framework_from_series, FrameWork, DataFrame, get_framework_from_dataframe

VarName = TypeVar('VarName')
Source = TypeVar('Source')
Target = TypeVar('Target')


def map_elements_series(series: Series, map_dict: Dict[Source, Target]) -> Series:
    framework = get_framework_from_series(series)
    func = _MAP_FRAMEWORK_map_elements_series[framework]
    return func(series, map_dict)


def _map_elements_series_pandas(series: Series, map_dict: Dict[Source, Target]) -> Series:
    series = series.map(lambda x: map_dict.get(x, x))
    return series


def _map_elements_series_polars(series: Series, map_dict: Dict[Source, Target]) -> Series:
    series = series.map_elements(lambda x: map_dict.get(x, x))
    return series


def _map_elements_series_pyspark(series: Series, map_dict: Dict[Source, Target]) -> Series:
    raise NotImplementedError


_MAP_FRAMEWORK_map_elements_series = {
    FrameWork.pandas: _map_elements_series_pandas,
    FrameWork.polars: _map_elements_series_polars,
    FrameWork.spark: _map_elements_series_pyspark,
}


def map_elements_df(df: DataFrame, map_dict: Dict[VarName, Dict[Source, Target]]) -> DataFrame:
    framework = get_framework_from_dataframe(df)
    func = _MAP_FRAMEWORK_map_elements_df[framework]
    return func(df, map_dict)


def _map_elements_df_pandas(df: DataFrame, map_dict: Dict[VarName, Dict[Source, Target]]) -> DataFrame:
    for col in map_dict:
        df[col] = df.map(lambda x: map_dict[col].get(x, x))
    return df


def _map_elements_df_polars(df: DataFrame, map_dict: Dict[VarName, Dict[Source, Target]]) -> DataFrame:
    import polars as pl

    for col_name, map_vals in map_dict.items():
        df = df.with_columns(pl.col(col_name).map_elements(lambda x: map_dict[col_name].get(x, x)).alias(col_name))

    # БАГ - вариант ниже не работает.
    # df = df.with_columns(*[
    #     pl.col(col_name).map_elements(lambda x: map_dict[col_name].get(x, x)).alias(col_name)
    #     for col_name in map_dict
    # ])
    return df


def _map_elements_df_pyspark(df: DataFrame, map_dict: Dict[VarName, Dict[Source, Target]]) -> DataFrame:
    raise NotImplementedError


_MAP_FRAMEWORK_map_elements_df = {
    FrameWork.pandas: _map_elements_df_pandas,
    FrameWork.polars: _map_elements_df_polars,
    FrameWork.spark: _map_elements_df_pyspark,
}


