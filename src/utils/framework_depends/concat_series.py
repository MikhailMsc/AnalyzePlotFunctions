from typing import List


from utils.general.types import Series, DataFrame, FrameWork, get_framework_from_series


def concat_series_to_frame(list_of_series: List[Series], columns: List[str]) -> DataFrame:
    framework = get_framework_from_series(list_of_series[0])
    func = _MAP_FRAMEWORK_FUNC[framework]
    return func(list_of_series, columns)


def _concat_series_to_frame_pandas(list_of_series: List[Series], columns: List[str]) -> DataFrame:
    import pandas as pd
    data = pd.DataFrame({col: ser for ser, col in zip(list_of_series, columns)})
    return data


def _concat_series_to_frame_polars(list_of_series: List[Series], columns: List[str]) -> DataFrame:
    import polars as pl
    data = pl.DataFrame({col: ser for ser, col in zip(list_of_series, columns)})
    return data


def _concat_series_to_frame_spark(list_of_series: List[Series], columns: List[str]) -> DataFrame:
    raise NotImplementedError


_MAP_FRAMEWORK_FUNC = {
    FrameWork.pandas: _concat_series_to_frame_pandas,
    FrameWork.polars: _concat_series_to_frame_polars,
    FrameWork.spark: _concat_series_to_frame_spark,
}
