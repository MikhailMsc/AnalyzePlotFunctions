from typing import List

from utils.general.types import DataFrame, get_framework_from_dataframe, FrameWork


def drop_columns(df: DataFrame, columns: List[str]) -> DataFrame:
    framework = get_framework_from_dataframe(df)
    func = _MAP_FRAMEWORK_FUNC[framework]
    return func(df, columns)


def _drop_columns_pandas(df: DataFrame, columns: List[str]) -> DataFrame:
    return df.drop(columns=columns)


def _drop_columns_polars(df: DataFrame, columns: List[str]) -> DataFrame:
    return df.drop(columns)


def _drop_columns_pyspark(df: DataFrame, columns: List[str]) -> DataFrame:
    raise df.drop(*columns)


_MAP_FRAMEWORK_FUNC = {
    FrameWork.pandas: _drop_columns_pandas,
    FrameWork.polars: _drop_columns_polars,
    FrameWork.spark: _drop_columns_pyspark,
}

