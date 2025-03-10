from utils.general.types import Series, get_framework_from_series, FrameWork


def get_count_missing(series: Series) -> int:
    framework = get_framework_from_series(series)
    func = _MAP_FRAMEWORK_FUNC[framework]
    return func(series)


def _get_count_missing_pandas(series: Series) -> int:
    return series.isnull().sum()


def _get_count_missing_polars(series: Series) -> int:
    return series.is_null().sum()


def _get_count_missing_pyspark(series: Series) -> int:
    raise NotImplementedError


_MAP_FRAMEWORK_FUNC = {
    FrameWork.pandas: _get_count_missing_pandas,
    FrameWork.polars: _get_count_missing_polars,
    FrameWork.spark: _get_count_missing_pyspark,
}







