from analyzer.utils.general.types import Series, get_framework_from_series, FrameWork


def len_series(series: Series) -> int:
    framework = get_framework_from_series(series)
    func = _MAP_FRAMEWORK_FUNC[framework]
    return func(series)


def _len_series_pandas(series: Series) -> int:
    return series.shape[0]


def _len_series_polars(series: Series) -> int:
    return series.shape[0]


def _len_series_pyspark(series: Series) -> int:
    raise NotImplementedError


_MAP_FRAMEWORK_FUNC = {
    FrameWork.pandas: _len_series_pandas,
    FrameWork.polars: _len_series_polars,
    FrameWork.spark: _len_series_pyspark,
}







