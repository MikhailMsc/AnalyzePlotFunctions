from utils.domain.columns import (
    C_GROUP, C_TOTAL, C_TARGET, C_POPULATION, C_TARGET_RATE, C_WOE,
    C_GROUP_IV, C_TOTAL_IV, C_TARGET_POPULATION
)
from utils.domain.const import MISSING
from utils.framework_depends import set_column, get_unique, convert_df_to_pandas, fill_missing_df, map_elements
from utils.framework_depends.columns.is_numeric_column import is_numeric_column
from utils.general.schema import SchemaDF
from utils.general.types import DataFrame, get_framework_from_dataframe, FrameWork
from preprocessing.binning import binarize_series


SH_InformationValueReport = SchemaDF(
    columns=[
        C_GROUP, C_TOTAL, C_TARGET, C_POPULATION, C_TARGET_POPULATION,
        C_TARGET_RATE, C_WOE, C_GROUP_IV, C_TOTAL_IV
    ],
    key=[C_GROUP]
)


def calc_information_value(
        df: DataFrame, var_name: str, target_name: str,
        need_binning: bool, map_values: dict = None, validate_target: bool = True, **binargs
) -> SH_InformationValueReport.t:
    """
    TODO: Дать подробное описание
    Args:
        df:
        var_name:
        target_name:
        need_binning:
        validate_target:
        **binargs:

    Returns:

    """
    if validate_target and get_unique(df[target_name]) != {0, 1}:
        raise Exception('Таргет должен содержать значения 0 и 1, пропуски не допускаются.')

    if need_binning:
        if not is_numeric_column(df[var_name]):
            raise Exception('Вы хотите применить алгоритм категоризации к нечисловой переменной.')

        var = binarize_series(variable=df[var_name], target=df[target_name], validate_target=False, **binargs)
        df = set_column(df, var, var_name)
    else:
        df = fill_missing_df(df, columns_values={var_name: MISSING})
        if map_values:
            map_elements(df[var_name], map_values)

    framework = get_framework_from_dataframe(df)
    func = _MAP_FRAMEWORK_FUNC[framework]
    iv_report = func(df, var_name, target_name)
    iv_report = SH_InformationValueReport(iv_report)
    return iv_report


def _calc_iv_pandas(df: DataFrame, var_name: str, target_name: str) -> DataFrame:
    import numpy as np
    df = df.groupby(var_name, as_index=False).agg({target_name: ['count', 'sum']})
    df.columns = [C_GROUP.n, C_TOTAL.n, C_TARGET.n]

    total_sum = df[C_TOTAL.n].sum()
    target_sum = df[C_TARGET.n].sum()
    not_target_sum = total_sum - target_sum

    df[C_POPULATION.n] = round(100 * df[C_TOTAL.n] / total_sum, 2)
    target_population = df[C_TARGET.n] / target_sum
    df[C_TARGET_POPULATION.n] = round(100 * target_population, 2)
    df[C_TARGET_RATE.n] = 100 * df[C_TARGET.n] / df[C_TOTAL.n]

    not_target = df[C_TOTAL.n] - df[C_TARGET.n]
    not_target_population = not_target / not_target_sum

    df[C_WOE.n] = np.log(target_population / not_target_population)
    df[C_GROUP_IV.n] = 100 * (target_population - not_target_population) * df[C_WOE.n]
    df[C_TOTAL_IV.n] = df[C_GROUP_IV.n].sum()
    return df


def _calc_iv_polars(df: DataFrame, var_name: str, target_name: str) -> DataFrame:
    import polars as pl

    df = df.group_by(var_name).agg(
        pl.col(target_name).count().alias(C_TOTAL.n),
        pl.col(target_name).sum().alias(C_TARGET.n)
    ).rename({var_name: C_GROUP.n})

    total_sum = df[C_TOTAL.n].sum()
    target_sum = df[C_TARGET.n].sum()
    not_target_sum = total_sum - target_sum

    target_population = df[C_TARGET.n] / target_sum

    df = df.with_columns(
        (100 * pl.col(C_TOTAL.n) / total_sum).round(2).alias(C_POPULATION.n),
        (100 * target_population).round(2).alias(C_TARGET_POPULATION.n),
        (100 * pl.col(C_TARGET.n) / pl.col(C_TOTAL.n)).alias(C_TARGET_RATE.n),
    )

    not_target = df[C_TOTAL.n] - df[C_TARGET.n]
    not_target_population = not_target / not_target_sum

    df = df.with_columns((target_population / not_target_population).log().alias(C_WOE.n))
    df = df.with_columns(
        (100 * (target_population - not_target_population) * pl.col(C_WOE.n)).alias(C_GROUP_IV.n)
    )
    df = df.with_columns(
        pl.col(C_GROUP_IV.n).sum().alias(C_TOTAL_IV.n)
    )
    df = df.sort(C_GROUP.n)
    df = convert_df_to_pandas(df)
    return df


def _calc_iv_spark(df: DataFrame, var_name: str, target_name: str) -> DataFrame:
    raise NotImplementedError


_MAP_FRAMEWORK_FUNC = {
    FrameWork.pandas: _calc_iv_pandas,
    FrameWork.polars: _calc_iv_polars,
    FrameWork.spark: _calc_iv_spark,
}
