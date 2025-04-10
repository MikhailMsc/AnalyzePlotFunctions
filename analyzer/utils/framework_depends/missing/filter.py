from functools import reduce
from typing import List

from analyzer.utils.framework_depends.columns import get_columns
from analyzer.utils.general.types import DataFrame, get_framework_from_dataframe, FrameWork


def filter_missing_df(df: DataFrame, columns: List[str] = None):
    framework = get_framework_from_dataframe(df)
    func = _MAP_FRAMEWORK_FUNC[framework]

    if columns is None:
        columns = get_columns(df)
    return func(df, columns)


def _filter_missing_df_pandas(df: DataFrame, columns: List[str]) -> DataFrame:
    filters = [~df[col].isnull() for col in columns]
    filters = reduce(lambda x, y: x & y, filters)
    df = df[filters].reset_index(drop=True)
    return df


def _filter_missing_df_polars(df: DataFrame, columns: List[str]) -> DataFrame:
    import polars as pl
    filters = [pl.col(col).is_not_null() for col in columns]
    df = df.filter(*filters)
    return df


def _filter_missing_df_pyspark(df: DataFrame, columns: List[str]) -> DataFrame:
    df = df.na.drop(subset=columns)
    return df


_MAP_FRAMEWORK_FUNC = {
    FrameWork.pandas: _filter_missing_df_pandas,
    FrameWork.polars: _filter_missing_df_polars,
    FrameWork.spark: _filter_missing_df_pyspark,
}






