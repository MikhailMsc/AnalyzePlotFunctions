from typing import List

from utils.general.types import DataFrame, get_framework_from_dataframe, FrameWork


def get_columns(df: DataFrame) -> List[str]:
    framework = get_framework_from_dataframe(df)
    func = _MAP_FRAMEWORK_FUNC[framework]
    return func(df)


def _get_columns_pandas(df: DataFrame) -> List[str]:
    return df.columns


def _get_columns_polars(df: DataFrame) -> List[str]:
    return df.columns


def _get_columns_pyspark(df: DataFrame) -> List[str]:
    return df.columns


_MAP_FRAMEWORK_FUNC = {
    FrameWork.pandas: _get_columns_pandas,
    FrameWork.polars: _get_columns_polars,
    FrameWork.spark: _get_columns_pyspark,
}


