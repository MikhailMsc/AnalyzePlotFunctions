from utils.general.types import DataFrame, Series, get_framework_from_dataframe, FrameWork


def set_column(df: DataFrame, series_or_value: Series, column: str):
    framework = get_framework_from_dataframe(df)
    func = _MAP_FRAMEWORK_FUNC[framework]
    return func(df, series_or_value, column)


def _set_column_pandas(df: DataFrame, series_or_value: Series, column: str) -> DataFrame:
    df[column] = series_or_value
    return df


def _set_column_polars(df: DataFrame, series_or_value: Series, column: str) -> DataFrame:
    import polars as pl
    if type(series_or_value) is pl.Series:
        df = df.with_columns(series_or_value.alias(column))
    else:
        df = df.with_columns(pl.lit(series_or_value).alias(column))
    return df


def _set_column_pyspark(df: DataFrame, series_or_value: Series, column: str) -> DataFrame:
    raise NotImplementedError


_MAP_FRAMEWORK_FUNC = {
    FrameWork.pandas: _set_column_pandas,
    FrameWork.polars: _set_column_polars,
    FrameWork.spark: _set_column_pyspark,
}


