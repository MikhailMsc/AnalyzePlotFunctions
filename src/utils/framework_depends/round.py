from utils.general.types import Series, get_framework_from_series, FrameWork


def round_series(series: Series, rnd: int) -> Series:
    framework = get_framework_from_series(series)
    func = _MAP_FRAMEWORK_FUNC[framework]
    return func(series, rnd)


def _round_series_pandas(series: Series, rnd: int) -> Series:
    series = series.round(rnd)
    return series


def _round_series_polars(series: Series, rnd: int) -> Series:
    series = series.round(rnd)
    return series


def _round_series_pyspark(series: Series, rnd: int) -> Series:
    series = series.round(rnd)
    return series


_MAP_FRAMEWORK_FUNC = {
    FrameWork.pandas: _round_series_pandas,
    FrameWork.polars: _round_series_polars,
    FrameWork.spark: _round_series_pyspark,
}






