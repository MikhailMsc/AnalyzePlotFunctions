from typing import List

from analyzer.utils.general.types import DataFrame, get_framework_from_dataframe, FrameWork


def reorder_columns(df: DataFrame, columns: List[str]) -> DataFrame:
    framework = get_framework_from_dataframe(df)
    func = _MAP_FRAMEWORK_FUNC[framework]
    return func(df, columns)


def _reorder_columns_pandas(df: DataFrame, columns: List[str]) -> DataFrame:
    return df[columns]


def _reorder_columns_polars(df: DataFrame, columns: List[str]) -> DataFrame:
    return df.select(columns)


def _reorder_columns_pyspark(df: DataFrame, columns: List[str]) -> DataFrame:
    raise NotImplementedError


_MAP_FRAMEWORK_FUNC = {
    FrameWork.pandas: _reorder_columns_pandas,
    FrameWork.polars: _reorder_columns_polars,
    FrameWork.spark: _reorder_columns_pyspark,
}

