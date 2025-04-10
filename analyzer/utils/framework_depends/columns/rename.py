from typing import Dict

from analyzer.utils.general.types import DataFrame, get_framework_from_dataframe, FrameWork


def rename_columns(df: DataFrame, rename_dict: Dict[str, str]) -> DataFrame:
    framework = get_framework_from_dataframe(df)
    func = _MAP_FRAMEWORK_FUNC[framework]
    return func(df, rename_dict)


def _rename_columns_pandas(df: DataFrame, rename_dict: Dict[str, str]) -> DataFrame:
    df.rename(columns=rename_dict, inplace=True)
    return df


def _rename_columns_polars(df: DataFrame, rename_dict: Dict[str, str]) -> DataFrame:
    return df.rename(rename_dict)


def _rename_columns_pyspark(df: DataFrame, rename_dict: Dict[str, str]) -> DataFrame:
    raise NotImplementedError


_MAP_FRAMEWORK_FUNC = {
    FrameWork.pandas: _rename_columns_pandas,
    FrameWork.polars: _rename_columns_polars,
    FrameWork.spark: _rename_columns_pyspark,
}

