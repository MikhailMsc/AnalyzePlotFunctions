from typing import List, Union, Tuple

import pandas as pd
from tqdm import tqdm

from analyzer import logger
from analyzer.preprocessing import BinningParamsMultiVars, MapDictMultiVars, preprocess_df, MapDictSingleVar
from analyzer.utils.domain.columns import (
    C_GROUP, C_TARGET, C_POPULATION, C_TARGET_RATE, C_WOE,
    C_GROUP_IV, C_TOTAL_IV, C_TARGET_POPULATION, C_VARNAME, C_GROUP_NUMBER, C_COUNT
)
from analyzer.utils.framework_depends import convert_df_to_pandas, get_columns

from analyzer.utils.general.schema import SchemaDF
from analyzer.utils.general.types import DataFrame, get_framework_from_dataframe, FrameWork
from analyzer.preprocessing.binning import BinningParams

SH_InformationValueReport = SchemaDF(
    columns=[
        C_VARNAME, C_GROUP_NUMBER, C_GROUP, C_COUNT, C_TARGET, C_POPULATION, C_TARGET_POPULATION,
        C_TARGET_RATE, C_GROUP_IV, C_TOTAL_IV
    ],
    key=[C_VARNAME, C_GROUP_NUMBER]
)

SH_ShortInformationValueReport = SchemaDF(columns=[C_VARNAME, C_TOTAL_IV], key=[C_VARNAME])

info_msg = f"""
    Var_Name - имя переменной,
    {C_GROUP_NUMBER.n} - номер категории,
    {C_GROUP.n} - значение категории,
    {C_COUNT.n} - общий размер категории,
    {C_TARGET.n} - количество таргетов в данной категории,
    {C_POPULATION.n} - относительный размер категории,
    {C_TARGET_POPULATION.n} - относительный размер таргета в данной категории,
    {C_TARGET_RATE.n} - Target Rate в данной категории,
    {C_GROUP_IV.n} - information value данной категории,
    {C_TOTAL_IV.n} - information value всей переменной
    
    Information Value (IV):
    - Чем больше значение ПО МОДУЛЮ = тем больше отклонение target rate oт среднего по выборке (сильнее разделяющая способность)
    - IV > 0 = target rate выше среднего по выборке
    - IV < 0 = target rate ниже среднего по выборке
"""


def calc_iv_report(
        df: DataFrame, target_name: str, analyze_vars: List[str] = None, ignore_vars: List[str] = None,
        map_values: MapDictMultiVars = None, binning: BinningParamsMultiVars = True,
        sort_by_iv: bool = True, _tqdm: bool = True, _logging: bool = True
) -> Tuple[SH_ShortInformationValueReport.t, SH_InformationValueReport.t]:
    """
    Формирует отчет Information Value для заданных переменных.

    Args:
        df:             Исследуемый датафрейм
        target_name:    Название таргета
        analyze_vars:   Список переменных для анализа, по умолчанию возьмутся все, кроме таргета и ignore_vars
        ignore_vars:    Список переменных игнорируемых во время анализа
        map_values:     Словарь для замены значений переменных (словарь, ключ = название переменной,
                        значение = словарь старое-новое значение)
        binning:        Параметры для биннинга, ниже будет более подробное описание.
        sort_by_iv:     Отсортировать по IV или по алфавиту.
        _tqdm:          Отображать прогрессбар
        _logging:       Вывод сообщения лога.

    Returns:
        DataFrame_1:
            - VARNAME
            - TOTAL_IV

        DataFrame_2:
            - VARNAME,
            - GROUP_NUMBER,
            - GROUP,
            - COUNT,
            - TARGET,
            - POPULATION,
            - TARGET_POPULATION,
            - TARGET_RATE,
            - GROUP_IV,
            - TOTAL_IV
    """
    df = preprocess_df(
        df, analyze_vars, ignore_vars, target_name, binning,
        map_values, _validate_target=True, drop_not_processed=True
    )
    analyze_vars = [col for col in get_columns(df) if col != target_name]
    if _tqdm:
        bar_format = (f"{{l_bar}}{{bar}}| Расчет IV отчета, {{n_fmt}}/{len(analyze_vars)} "
                      f"[{{elapsed}}<{{remaining}}, {{rate_fmt}}]")
        iter_obj = tqdm(analyze_vars, total=len(analyze_vars), bar_format=bar_format)
    else:
        iter_obj = analyze_vars

    total_report = []
    for var_name in iter_obj:
        single_report = calc_iv_var(
            df=df, var_name=var_name, target_name=target_name,
            binning=False,
            _validate_target=False,
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

    if _logging:
        logger.log_info(info_msg)
    return total_report_short, total_report


def calc_iv_var(
        df: DataFrame, var_name: str, target_name: str, map_values: MapDictSingleVar = None,
        binning: Union[BinningParams, bool] = True, _validate_target: bool = True
) -> SH_InformationValueReport.t:
    """
    Расчет Information Value для одной переменной.

    Args:
        df:                 Исследуемый датафрейм
        var_name:           Название переменной
        target_name         Название таргета
        map_values:         Мэппинг значений переменной
        binning:            Параметры для биннинга или булев флаг (нужен или нет биннинг)
        _validate_target:   Скрытый параметр, валидация бинарности таргета

    Returns:
        DataFrame:
            - VARNAME,
            - GROUP_NUMBER,
            - GROUP,
            - COUNT,
            - TARGET,
            - POPULATION,
            - TARGET_POPULATION,
            - TARGET_RATE,
            - GROUP_IV,
            - TOTAL_IV
    """
    if _validate_target or binning or map_values:
        if map_values is not None:
            map_values = {var_name: map_values}

        df = preprocess_df(
            df, [var_name], target_name=target_name, binning=binning,
            map_values=map_values, _validate_target=_validate_target, drop_not_processed=True,
            _tqdm=False
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
    df = df.groupby(var_name, as_index=False, observed=True).agg({target_name: ['size', 'sum']})
    df.columns = [C_GROUP.n, C_COUNT.n, C_TARGET.n]

    total_sum = df[C_COUNT.n].sum()
    target_sum = df[C_TARGET.n].sum()
    not_target_sum = total_sum - target_sum

    df[C_POPULATION.n] = round(100 * df[C_COUNT.n] / total_sum, 2)
    target_population = df[C_TARGET.n] / target_sum
    df[C_TARGET_POPULATION.n] = round(100 * target_population, 2)
    df[C_TARGET_RATE.n] = 100 * df[C_TARGET.n] / df[C_COUNT.n]

    not_target = df[C_COUNT.n] - df[C_TARGET.n]
    not_target_population = not_target / not_target_sum

    target_population.replace(0, 1 / target_sum, inplace=True)
    not_target_population.replace(0, 1 / not_target_sum, inplace=True)

    df[C_WOE.n] = np.log(target_population / not_target_population)
    df[C_GROUP_IV.n] = 100 * (target_population - not_target_population).abs() * df[C_WOE.n]
    df[C_TOTAL_IV.n] = df[C_GROUP_IV.n].abs().sum()
    return df


def _calc_iv_polars(df: DataFrame, var_name: str, target_name: str) -> DataFrame:
    import polars as pl

    df = df.group_by(var_name).agg(
        pl.col(target_name).len().alias(C_COUNT.n),
        pl.col(target_name).sum().alias(C_TARGET.n)
    ).rename({var_name: C_GROUP.n})

    total_sum = df[C_COUNT.n].sum()
    target_sum = df[C_TARGET.n].sum()
    not_target_sum = total_sum - target_sum

    target_population = df[C_TARGET.n] / target_sum

    df = df.with_columns(
        (100 * pl.col(C_COUNT.n) / total_sum).round(2).alias(C_POPULATION.n),
        (100 * target_population).round(2).alias(C_TARGET_POPULATION.n),
        (100 * pl.col(C_TARGET.n) / pl.col(C_COUNT.n)).alias(C_TARGET_RATE.n),
    )

    not_target = df[C_COUNT.n] - df[C_TARGET.n]
    not_target_population = not_target / not_target_sum

    target_population = target_population.replace(0, 1 / target_sum)
    not_target_population = not_target_population.replace(0, 1 / not_target_sum)

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
