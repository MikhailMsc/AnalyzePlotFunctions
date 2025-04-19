from typing import List

from analyzer.utils.general.types import Series, get_framework_from_series, FrameWork


def series_to_list(ser: Series) -> List:
    framework = get_framework_from_series(ser)
    func = _MAP_FRAMEWORK_FUNC[framework]
    return func(ser)


def _series_to_list_pandas(ser: Series) -> List:
    return ser.to_list()


def _series_to_list_polars(ser: Series) -> List:
    return ser.to_list()


_MAP_FRAMEWORK_FUNC = {
    FrameWork.pandas: _series_to_list_pandas,
    FrameWork.polars: _series_to_list_polars,
    # FrameWork.spark: _convert_df_from_spark_to_pandas,
}
