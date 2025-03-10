from utils.general.types import Series, get_framework_from_series, FrameWork


def get_max(series: Series):
    framework = get_framework_from_series(series)
    func = _MAP_FRAMEWORK_FUNC[framework]
    return func(series)


def _get_max_pandas(series: Series):
    return series.max()


def _get_max_polars(series: Series):
    return series.max()


def _get_max_pyspark(series: Series):
    return series.max()


_MAP_FRAMEWORK_FUNC = {
    FrameWork.pandas: _get_max_pandas,
    FrameWork.polars: _get_max_polars,
    FrameWork.spark: _get_max_pyspark,
}

