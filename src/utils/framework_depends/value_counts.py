from utils.domain.columns import C_VALUE, C_COUNT, C_PERCENT
from utils.general.schema import SchemaDF
from utils.general.types import Series, DataFrame, get_framework_from_series, FrameWork

SH_ValueCounts = SchemaDF(
    columns=[C_VALUE, C_COUNT, C_PERCENT],
    key=[C_VALUE],
    name='ValueCounts'
)


def value_counts(series: Series, sort: bool):
    framework = get_framework_from_series(series)
    func = _MAP_FRAMEWORK_FUNC[framework]
    return func(series, sort)


def _value_counts_pandas(series: Series, sort: bool) -> DataFrame:
    stats = series.value_counts(dropna=False)
    if sort:
        stats = stats.sort_index(ascending=True).reset_index()
    stats[C_PERCENT.n] = 100 * stats[C_COUNT.n] / stats[C_COUNT.n].sum()
    stats.columns = [C_VALUE.n, C_COUNT.n, C_PERCENT.n]
    return stats


def _value_counts_polars(series: Series, sort: bool) -> DataFrame:
    stats = series.value_counts(parallel=True)
    stats.columns = [C_VALUE.n, C_COUNT.n]

    if sort:
        stats = stats.sort(C_VALUE.n, descending=False)

    stats = stats.with_columns((100 * stats[C_COUNT.n] / stats[C_COUNT.n].sum()).alias(C_PERCENT.n))
    return stats


def _value_counts_pyspark(series: Series, sort: bool) -> DataFrame:
    raise NotImplementedError


_MAP_FRAMEWORK_FUNC = {
    FrameWork.pandas: _value_counts_pandas,
    FrameWork.polars: _value_counts_polars,
    FrameWork.spark: _value_counts_pyspark,
}

