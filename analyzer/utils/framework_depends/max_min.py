from analyzer.utils.general.types import Series, get_framework_from_series, FrameWork


def get_max(series: Series):
    framework = get_framework_from_series(series)
    func = _MAP_FRAMEWORK_get_max[framework]
    return func(series)


def _get_max_pandas(series: Series):
    return series.max()


def _get_max_polars(series: Series):
    return series.max()


def _get_max_pyspark(series: Series):
    return series.max()


_MAP_FRAMEWORK_get_max = {
    FrameWork.pandas: _get_max_pandas,
    FrameWork.polars: _get_max_polars,
    FrameWork.spark: _get_max_pyspark,
}


def get_min(series: Series):
    framework = get_framework_from_series(series)
    func = _MAP_FRAMEWORK_get_min[framework]
    return func(series)


def _get_min_pandas(series: Series):
    return series.min()


def _get_min_polars(series: Series):
    return series.min()


def _get_min_pyspark(series: Series):
    return series.min()


_MAP_FRAMEWORK_get_min = {
    FrameWork.pandas: _get_min_pandas,
    FrameWork.polars: _get_min_polars,
    FrameWork.spark: _get_min_pyspark,
}
