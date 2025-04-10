from typing import Tuple

from analyzer.utils.general.types import DataFrame, get_framework_from_dataframe, FrameWork


def get_shape(df: DataFrame) -> Tuple[int, int]:
    framework = get_framework_from_dataframe(df)
    func = _MAP_FRAMEWORK_FUNC[framework]
    return func(df)


def _get_shape_pandas(df: DataFrame) -> Tuple[int, int]:
    return df.shape


def _get_shape_polars(df: DataFrame) -> Tuple[int, int]:
    return df.shape


def _get_shape_pyspark(df: DataFrame) -> Tuple[int, int]:
    return df.shape


_MAP_FRAMEWORK_FUNC = {
    FrameWork.pandas: _get_shape_pandas,
    FrameWork.polars: _get_shape_polars,
    FrameWork.spark: _get_shape_pyspark,
}
