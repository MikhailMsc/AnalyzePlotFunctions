from typing import List

from analyzer.utils.general.types import DataFrame, get_framework_from_dataframe, FrameWork, Series


def get_columns(df: DataFrame) -> List[str]:
    framework = get_framework_from_dataframe(df)
    func = _MAP_FRAMEWORK_FUNC_get_columns[framework]
    return func(df)


def _get_columns_pandas(df: DataFrame) -> List[str]:
    return df.columns


def _get_columns_polars(df: DataFrame) -> List[str]:
    return df.columns


def _get_columns_pyspark(df: DataFrame) -> List[str]:
    return df.columns


_MAP_FRAMEWORK_FUNC_get_columns = {
    FrameWork.pandas: _get_columns_pandas,
    FrameWork.polars: _get_columns_polars,
    FrameWork.spark: _get_columns_pyspark,
}


def get_sub_df(df: DataFrame, columns: List[str]) -> DataFrame:
    framework = get_framework_from_dataframe(df)
    func = _MAP_FRAMEWORK_FUNC_get_sub_df[framework]
    return func(df, columns)


def _get_sub_df_pandas(df: DataFrame, columns: List[str]) -> DataFrame:
    return df[columns]


def _get_sub_df_polars(df: DataFrame, columns: List[str]) -> DataFrame:
    return df[columns]


def _get_sub_df_pyspark(df: DataFrame, columns: List[str]) -> DataFrame:
    return df[columns]


_MAP_FRAMEWORK_FUNC_get_sub_df = {
    FrameWork.pandas: _get_sub_df_pandas,
    FrameWork.polars: _get_sub_df_polars,
    FrameWork.spark: _get_sub_df_pyspark,
}


def get_series_from_df(df: DataFrame, column: str) -> Series:
    framework = get_framework_from_dataframe(df)
    func = _MAP_FRAMEWORK_FUNC_get_series_from_df[framework]
    return func(df, column)


def _get_series_from_df_pandas(df: DataFrame, column: str) -> Series:
    return df[column]


def _get_series_from_df_polars(df: DataFrame, column: str) -> Series:
    return df[column]


def _get_series_from_df_pyspark(df: DataFrame, column: str) -> Series:
    return df[column]


_MAP_FRAMEWORK_FUNC_get_series_from_df = {
    FrameWork.pandas: _get_series_from_df_pandas,
    FrameWork.polars: _get_series_from_df_polars,
    FrameWork.spark: _get_series_from_df_pyspark,
}

