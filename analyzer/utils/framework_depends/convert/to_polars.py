from analyzer.utils.general.types import DataFrame, get_framework_from_dataframe, FrameWork


def convert_df_to_polars(df: DataFrame) -> DataFrame:
    framework = get_framework_from_dataframe(df)
    if framework == FrameWork.polars:
        return df
    else:
        func = _MAP_FRAMEWORK_FUNC[framework]
        return func(df)


def _convert_df_from_pandas_to_polars(df: DataFrame) -> DataFrame:
    import polars as pl
    return pl.from_pandas(df)


def _convert_df_from_spark_to_polars(df: DataFrame) -> DataFrame:
    return _convert_df_from_pandas_to_polars(df.toPandas())


_MAP_FRAMEWORK_FUNC = {
    FrameWork.pandas: _convert_df_from_pandas_to_polars,
    FrameWork.spark: _convert_df_from_spark_to_polars,
}



