from utils.general.types import DataFrame, Series, get_framework_from_dataframe, FrameWork


def set_column(df: DataFrame, series: Series, column: str):
    framework = get_framework_from_dataframe(df)
    func = _MAP_FRAMEWORK_FUNC[framework]
    return func(df, series, column)


def _set_column_pandas(df: DataFrame, series: Series, column: str) -> DataFrame:
    df[column] = series
    return df


def _set_column_polars(df: DataFrame, series: Series, column: str) -> DataFrame:
    df = df.with_columns(series.alias(column))
    return df


def _set_column_pyspark(df: DataFrame, series: Series, column: str) -> DataFrame:
    raise NotImplementedError


_MAP_FRAMEWORK_FUNC = {
    FrameWork.pandas: _set_column_pandas,
    FrameWork.polars: _set_column_polars,
    FrameWork.spark: _set_column_pyspark,
}


