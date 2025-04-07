from typing import List, Union, Tuple

import pandas as pd

from preprocessing import BinningParamsMultiVars, MapDictMultiVars, preprocess_df, MapDictSingleVar
from utils.domain.columns import (
    C_GROUP, C_TOTAL, C_TARGET, C_POPULATION, C_TARGET_RATE, C_WOE,
    C_GROUP_IV, C_TOTAL_IV, C_TARGET_POPULATION, C_VARNAME, C_GROUP_NUMBER
)
from utils.framework_depends import convert_df_to_pandas

from utils.general.schema import SchemaDF
from utils.general.types import DataFrame, get_framework_from_dataframe, FrameWork
from preprocessing.binning import BinningParams

SH_InformationValueReport = SchemaDF(
    columns=[
        C_VARNAME, C_GROUP_NUMBER, C_GROUP, C_TOTAL, C_TARGET, C_POPULATION, C_TARGET_POPULATION,
        C_TARGET_RATE, C_GROUP_IV, C_TOTAL_IV
    ],
    key=[C_VARNAME, C_GROUP_NUMBER]
)

SH_ShortInformationValueReport = SchemaDF(columns=[C_VARNAME, C_TOTAL_IV], key=[C_VARNAME])


def calc_iv_report(
        df: DataFrame, target_name: str, analyze_vars: List[str] = None, ignore_vars: List[str] = None,
        binning: BinningParamsMultiVars = True,
        map_values: MapDictMultiVars = None, sort_by_iv: bool = True
) -> Tuple[SH_ShortInformationValueReport.t, SH_InformationValueReport.t]:

    df = preprocess_df(
        df, analyze_vars, ignore_vars, target_name, binning,
        map_values, True, True
    )

    total_report = []
    for var_name in analyze_vars:
        single_report = calc_iv_var(
            df=df, var_name=var_name, target_name=target_name,
            binning=False,
            validate_target=False,
        )
        total_report.append(single_report)

    total_report = pd.concat(total_report, axis=0)

    if sort_by_iv:
        total_report = total_report.sort_values([C_TOTAL_IV.n, C_GROUP_NUMBER.n], ascending=[False, True])
    else:
        total_report = total_report.sort_values([C_VARNAME.n, C_GROUP_NUMBER.n], ascending=[True, True])

    total_report.reset_index(drop=True, inplace=True)
    total_report_short = (
        total_report[SH_ShortInformationValueReport.col_names].
        drop_duplicates().
        reset_index(drop=True)
    )

    return total_report_short, total_report


def calc_iv_var(
        df: DataFrame, var_name: str, target_name: str,
        binning: Union[BinningParams, bool] = True, map_values: MapDictSingleVar = None,
        validate_target: bool = True
) -> SH_InformationValueReport.t:
    """
    TODO: Дать подробное описание
    """
    if validate_target or binning or map_values:
        if map_values is not None:
            map_values = {var_name: map_values}

        df = preprocess_df(
            df, [var_name], target_name=target_name, binning=binning,
            map_values=map_values, validate_target=validate_target, drop_not_processed=True
        )

    framework = get_framework_from_dataframe(df)
    func = _MAP_FRAMEWORK_FUNC[framework]
    iv_report = func(df, var_name, target_name)

    iv_report[C_VARNAME.n] = var_name
    iv_report[C_GROUP_NUMBER.n] = iv_report.index
    iv_report[C_GROUP_NUMBER.n] += 1

    iv_report = SH_InformationValueReport(iv_report, reorder_colums=True)
    return iv_report


def _calc_iv_pandas(df: DataFrame, var_name: str, target_name: str) -> DataFrame:
    import numpy as np
    df = df.groupby(var_name, as_index=False).agg({target_name: ['size', 'sum']})
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
    df[C_GROUP_IV.n] = 100 * (target_population - not_target_population).abs() * df[C_WOE.n]
    df[C_TOTAL_IV.n] = df[C_GROUP_IV.n].abs().sum()
    return df


def _calc_iv_polars(df: DataFrame, var_name: str, target_name: str) -> DataFrame:
    import polars as pl

    df = df.group_by(var_name).agg(
        pl.col(target_name).len().alias(C_TOTAL.n),
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
        (100 * (target_population - not_target_population).abs() * pl.col(C_WOE.n)).alias(C_GROUP_IV.n)
    )
    df = df.with_columns(
        pl.col(C_GROUP_IV.n).abs().sum().alias(C_TOTAL_IV.n)
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
