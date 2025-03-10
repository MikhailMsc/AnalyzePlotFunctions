from utils.general.types import Series, get_framework_from_series, FrameWork


def is_numeric_column(series: Series) -> bool:
    framework = get_framework_from_series(series)
    func = _MAP_FRAMEWORK_FUNC[framework]
    return func(series)


def _is_numeric_column_pandas(series: Series) -> bool:
    dtype = str(series.dtype).lower()
    is_numeric = ('int' in dtype) or ('float' in dtype) or ('decimal' in dtype)
    return is_numeric


def _is_numeric_column_polars(series: Series) -> bool:
    dtype = str(series.dtype).lower()
    is_numeric = ('int' in dtype) or ('float' in dtype) or ('decimal' in dtype)
    return is_numeric


def _is_numeric_column_spark(series: Series) -> bool:
    raise NotImplementedError


_MAP_FRAMEWORK_FUNC = {
    FrameWork.pandas: _is_numeric_column_pandas,
    FrameWork.polars: _is_numeric_column_polars,
    FrameWork.spark: _is_numeric_column_spark,
}
