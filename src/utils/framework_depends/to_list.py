from utils.general.types import Series, get_framework_from_series, FrameWork


def series_to_list(series: Series) -> list:
    framework = get_framework_from_series(series)
    func = _MAP_FRAMEWORK_FUNC[framework]
    return func(series)


def _series_to_list_pandas(series: Series) -> list:
    return series.to_list()


def _series_to_list_polars(series: Series) -> list:
    return series.to_list()


def _series_to_list_pyspark(series: Series) -> list:
    return series.to_list()


_MAP_FRAMEWORK_FUNC = {
    FrameWork.pandas: _series_to_list_pandas,
    FrameWork.polars: _series_to_list_polars,
    FrameWork.spark: _series_to_list_pyspark,
}


