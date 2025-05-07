from typing import Dict, Any

from analyzer.utils.general.types import DataFrame, get_framework_from_dataframe, FrameWork


def fill_missing_df(df: DataFrame, columns_values: Dict[str, Any]) -> DataFrame:
    framework = get_framework_from_dataframe(df)
    func = _MAP_FRAMEWORK_FUNC[framework]
    return func(df, columns_values)


def _fill_missing_df_pandas(df: DataFrame, columns_values: Dict[str, Any]) -> DataFrame:
    for col in df.columns:
        if col not in columns_values:
            continue

        if df[col].dtype.name == 'category':
            if df[col].isnull().sum() > 0:
                df[col] = df[col].cat.add_categories([columns_values[col]])
            else:
                del columns_values[col]

    return df.fillna(columns_values)


def _fill_missing_df_polars(df: DataFrame, columns_values: Dict[str, Any]) -> DataFrame:
    import polars as pl
    df = df.with_columns(
        pl.col(col).fill_null(val).alias(col) for col, val in columns_values.items()
    )
    return df


def _fill_missing_df_pyspark(df: DataFrame, columns_values: Dict[str, Any]) -> DataFrame:
    raise NotImplementedError


_MAP_FRAMEWORK_FUNC = {
    FrameWork.pandas: _fill_missing_df_pandas,
    FrameWork.polars: _fill_missing_df_polars,
    FrameWork.spark: _fill_missing_df_pyspark,
}

