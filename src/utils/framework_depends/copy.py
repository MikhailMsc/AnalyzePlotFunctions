from utils.general.types import get_framework_from_dataframe, FrameWork, DataFrame


def copy_df(df: DataFrame) -> DataFrame:
    framework = get_framework_from_dataframe(df)
    func = _MAP_FRAMEWORK_FUNC[framework]
    return func(df)


def _copy_df_pandas(df: DataFrame) -> DataFrame:
    return df.copy(deep=True)


def _copy_df_polars(df: DataFrame) -> DataFrame:
    return df.clone()


def _copy_df_pyspark(df: DataFrame) -> DataFrame:
    raise NotImplementedError


_MAP_FRAMEWORK_FUNC = {
    FrameWork.pandas: _copy_df_pandas,
    FrameWork.polars: _copy_df_polars,
    FrameWork.spark: _copy_df_pyspark,
}



