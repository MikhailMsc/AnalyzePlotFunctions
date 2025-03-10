from utils.general.types import Series, get_framework_from_series, FrameWork


def get_unique(series: Series) -> set:
    framework = get_framework_from_series(series)
    func = _MAP_FRAMEWORK_FUNC[framework]
    return func(series)


def _get_unique_pandas(series: Series) -> set:
    return set(series.unique().tolist())


def _get_unique_polars(series: Series) -> set:
    return set(series.unique().to_list())


def _get_unique_pyspark(series: Series) -> set:
    raise NotImplementedError
    # return series.unique()


_MAP_FRAMEWORK_FUNC = {
    FrameWork.pandas: _get_unique_pandas,
    FrameWork.polars: _get_unique_polars,
    FrameWork.spark: _get_unique_pyspark,
}

