from utils.general.types import Series, get_framework_from_series, FrameWork


def is_numeric_column(series: Series) -> bool:
    framework = get_framework_from_series(series)
    func = _MAP_FRAMEWORK_is_numeric_column[framework]
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


_MAP_FRAMEWORK_is_numeric_column = {
    FrameWork.pandas: _is_numeric_column_pandas,
    FrameWork.polars: _is_numeric_column_polars,
    FrameWork.spark: _is_numeric_column_spark,
}


def is_integer_column(series: Series) -> bool:
    framework = get_framework_from_series(series)
    func = _MAP_FRAMEWORK_is_numeric_column[framework]
    return func(series)


def _is_integer_column_pandas(series: Series) -> bool:
    dtype = str(series.dtype).lower()
    is_numeric = 'int' in dtype
    return is_numeric


def _is_integer_column_polars(series: Series) -> bool:
    dtype = str(series.dtype).lower()
    is_numeric = 'int' in dtype
    return is_numeric


def _is_integer_column_spark(series: Series) -> bool:
    raise NotImplementedError


_MAP_FRAMEWORK_is_integer_column = {
    FrameWork.pandas: _is_integer_column_pandas,
    FrameWork.polars: _is_integer_column_polars,
    FrameWork.spark: _is_integer_column_spark,
}


def is_convertable_to_int_column(series: Series) -> bool:
    framework = get_framework_from_series(series)
    func = _MAP_FRAMEWORK_is_numeric_column[framework]
    return func(series)


def _is_convertable_to_int_column_pandas(series: Series) -> bool:
    return (series % 1).sum() == 0


def _is_convertable_to_int_column_polars(series: Series) -> bool:
    return (series % 1).sum() == 0


def _is_convertable_to_int_column_spark(series: Series) -> bool:
    raise NotImplementedError


_MAP_FRAMEWORK_is_convertable_to_int_column = {
    FrameWork.pandas: _is_convertable_to_int_column_pandas,
    FrameWork.polars: _is_convertable_to_int_column_polars,
    FrameWork.spark: _is_convertable_to_int_column_spark,
}
