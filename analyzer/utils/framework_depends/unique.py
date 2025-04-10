from typing import List, Dict

from analyzer.utils.general.types import (
    Series, get_framework_from_series, FrameWork, DataFrame, get_framework_from_dataframe
)


def get_unique(series: Series) -> set:
    framework = get_framework_from_series(series)
    func = _MAP_FRAMEWORK_get_unique[framework]
    return func(series)


def _get_unique_pandas(series: Series) -> set:
    return set(series.unique().tolist())


def _get_unique_polars(series: Series) -> set:
    return set(series.unique().to_list())


def _get_unique_pyspark(series: Series) -> set:
    raise NotImplementedError
    # return series.unique()


_MAP_FRAMEWORK_get_unique = {
    FrameWork.pandas: _get_unique_pandas,
    FrameWork.polars: _get_unique_polars,
    FrameWork.spark: _get_unique_pyspark,
}


def get_nunique(df: DataFrame, columns: List[str] = None) -> Dict[str, int]:
    framework = get_framework_from_dataframe(df)
    func = _MAP_FRAMEWORK_get_nunique[framework]
    return func(df, columns)


def _get_nunique_pandas(df: DataFrame, columns: List[str] = None) -> Dict[str, int]:
    if columns is None:
        columns = df.columns
    return df[columns].nunique().to_dict()


def _get_nunique_polars(df: DataFrame, columns: List[str] = None) -> Dict[str, int]:
    import polars as pl

    if columns is not None:
        df = df.select(columns)
    df = df.select(pl.all().n_unique()).to_dicts()[0]
    return df


def _get_nunique_pyspark(df: DataFrame, columns: List[str] = None) -> Dict[str, int]:
    raise NotImplementedError
    # return series.unique()


_MAP_FRAMEWORK_get_nunique = {
    FrameWork.pandas: _get_nunique_pandas,
    FrameWork.polars: _get_nunique_polars,
    FrameWork.spark: _get_nunique_pyspark,
}