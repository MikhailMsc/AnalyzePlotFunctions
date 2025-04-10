from typing import List


from analyzer.utils.general.types import (
    Series, DataFrame, FrameWork, get_framework_from_series, get_framework_from_dataframe
)


def concat_series_to_frame(list_of_series: List[Series], columns: List[str]) -> DataFrame:
    framework = get_framework_from_series(list_of_series[0])
    func = _MAP_FRAMEWORK_concat_series_to_frame[framework]
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


_MAP_FRAMEWORK_concat_series_to_frame = {
    FrameWork.pandas: _concat_series_to_frame_pandas,
    FrameWork.polars: _concat_series_to_frame_polars,
    FrameWork.spark: _concat_series_to_frame_spark,
}


def concat_df(list_of_dfs: List[DataFrame], horizontal: bool = False, vertical: bool = False) -> DataFrame:
    assert horizontal or vertical
    framework = get_framework_from_dataframe(list_of_dfs[0])
    func = _MAP_FRAMEWORK_concat_df[framework]
    return func(list_of_dfs, horizontal, vertical)


def _concat_df_pandas(list_of_dfs: List[DataFrame], horizontal: bool = False, vertical: bool = False) -> DataFrame:
    import pandas as pd
    if horizontal:
        axis = 0
    else:
        axis = 1
    data = pd.concat(list_of_dfs, axis=axis, ignore_index=True)
    return data


def _concat_df_polars(list_of_dfs: List[DataFrame], horizontal: bool = False, vertical: bool = False) -> DataFrame:
    import polars as pl
    how = 'horizontal' if horizontal else 'diagonal'
    data = pl.concat(list_of_dfs, how=how)
    return data


def _concat_df_spark(list_of_dfs: List[DataFrame], horizontal: bool = False, vertical: bool = False) -> DataFrame:
    raise NotImplementedError


_MAP_FRAMEWORK_concat_df = {
    FrameWork.pandas: _concat_df_pandas,
    FrameWork.polars: _concat_df_polars,
    FrameWork.spark: _concat_df_spark,
}
