from utils.general.types import DataFrame, get_framework_from_dataframe, FrameWork


def convert_df_to_pandas(df: DataFrame) -> DataFrame:
    framework = get_framework_from_dataframe(df)
    if framework == FrameWork.pandas:
        return df
    else:
        func = _MAP_FRAMEWORK_FUNC[framework]
        return func(df)


def _convert_df_from_polars_to_pandas(df: DataFrame) -> DataFrame:
    return df.to_pandas()


def _convert_df_from_spark_to_pandas(df: DataFrame) -> DataFrame:
    return df.toPandas()


_MAP_FRAMEWORK_FUNC = {
    FrameWork.polars: _convert_df_from_polars_to_pandas,
    FrameWork.spark: _convert_df_from_spark_to_pandas,
}






