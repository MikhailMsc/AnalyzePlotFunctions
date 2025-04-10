from enum import Enum
from typing import TypeVar

Series = TypeVar('Series')
DataFrame = TypeVar('DataFrame')


class FrameWork(Enum):
    pandas = 'pandas'
    polars = 'polars'
    spark = 'pyspark'


def get_framework_from_series(series: Series) -> FrameWork:
    import pandas as pd
    if type(series) is pd.Series:
        return FrameWork.pandas

    import polars as pl
    if type(series) is pl.Series:
        return FrameWork.polars

    import pyspark.pandas as ps
    if type(series) is ps.Series:
        return FrameWork.spark

    raise Exception('Используется неизвестный фреймворк отвечающий за данные (DataFrame, Series)')


def get_framework_from_dataframe(df: DataFrame) -> FrameWork:
    import pandas as pd
    if type(df) is pd.DataFrame:
        return FrameWork.pandas

    import polars as pl
    if type(df) is pl.DataFrame:
        return FrameWork.polars

    import pyspark.pandas as ps
    if type(df) is ps.DataFrame:
        return FrameWork.spark

    raise Exception('Используется неизвестный фреймворк отвечающий за данные (DataFrame, Series)')

