"""
Что будем считать:
- стабильность популяции группы
- стабильность популяции таргета = Сохраняем ли пропорции таргета между группами.
- стабильность


Total Report:
- SplitColumn
- Cnt Total
- Cnt Target
- TotalPopulationChange = Статистика общего изменения количества наблюдений в выборке
- TotalTargetStability = Стабильность распределения таргета между выборками.

Detail Report:
- SplitColumn
- Переменная
- Группа/Значение
- Base_poplulation_0, %
- Split_population_0, %
- Base_poplulation_1, %
- Split_population_1, %
- Stability_0
- Stability_1


0.8 0.7
0.2 0.3


(0.8 - 0.7) * np.log(0.8 / 0.7) = 0.013353139262452289
(0.7 - 0.8) * np.log(0.7 / 0.8) = 0.013353139262452289

(0.3 - 0.2) * np.log(0.3 / 0.2) = 0.013353139262452289
"""
from typing import Union, Dict, Any

import pandas as pd
import numpy as np

from utils.domain.columns import (
    C_COUNT_MAIN, C_COUNT_SECONDARY, C_POPULATION_MAIN, C_POPULATION_SECONDARY,
    C_TARGET_MAIN, C_TARGET_SECONDARY, C_TARGET_RATE_MAIN, C_TARGET_RATE_SECONDARY,
    C_TARGET_POPULATION_MAIN, C_TARGET_POPULATION_SECONDARY, C_POPULATION_STABILITY,
    C_TARGET_STABILITY, C_VARNAME, C_VALUE, C_SPLIT_COLUMN
)
from utils.domain.const import MISSING
from utils.domain.validate import validate_binary_target
from utils.framework_depends import (
    get_series_from_df, get_sub_df, set_column, fill_missing_df,
    map_elements_series, get_unique, rename_columns, get_columns
)
from utils.general.column import Column
from utils.general.schema import SchemaDF
from utils.general.types import DataFrame, FrameWork, get_framework_from_dataframe

from preprocessing.binning import binarize_series, BinningParams



SH_StabilityReportNoTarget = SchemaDF(
    columns=[
        C_SPLIT_COLUMN, C_VARNAME, C_VALUE,
        C_COUNT_MAIN, C_COUNT_SECONDARY, C_POPULATION_MAIN, C_POPULATION_SECONDARY,
        C_POPULATION_STABILITY,
    ],
    key=[C_SPLIT_COLUMN, C_VARNAME, C_VALUE]
)

SH_StabilityReportWithTarget = SchemaDF(
    columns=[
        C_SPLIT_COLUMN, C_VARNAME, C_VALUE,
        C_COUNT_MAIN, C_COUNT_SECONDARY, C_POPULATION_MAIN, C_POPULATION_SECONDARY,
        C_TARGET_MAIN, C_TARGET_SECONDARY, C_TARGET_RATE_MAIN, C_TARGET_RATE_SECONDARY,
        C_TARGET_POPULATION_MAIN, C_TARGET_POPULATION_SECONDARY,
        C_POPULATION_STABILITY, C_TARGET_STABILITY,
    ],
    key=[C_SPLIT_COLUMN, C_VARNAME, C_VALUE]
)

# SH_ShortInformationValueReport = SchemaDF(columns=[C_VARNAME, C_TOTAL_IV], key=[C_VARNAME])


c_secondary_count_total = 'secondary_count_total'
c_secondary_target_total = 'secondary_target_total'



def calc_all_stability():
    print(f"""
        {C_SPLIT_COLUMN}.n - поле, которое разбивает данные на подвыборки для сверки их характеристик между собой,
        {C_VARNAME.n}_[i] - имя i-ой переменной участвующей в сегменте, 
        {C_VALUE.n}_[i] - значение i-ой переменной участвующей в сегменте, 
        {C_COUNT_MAIN.n} - количество наблюдений в данном сегменте базовой выборки (относительно которой будем сверять),
        {C_COUNT_SECONDARY.n} - количество наблюдений в данном сегменте в сверяемой выборке (определяется по {C_SPLIT_COLUMN}.n),
        {C_POPULATION_MAIN.n} - относительный размер сегмента в базовой выборке, 
        {C_POPULATION_SECONDARY.n} - относительный размер сегмента в сверяемой выборке, 
        {C_TARGET_MAIN.n} - количество таргетов в данном сегменте базовой выборки, 
        {C_TARGET_SECONDARY.n} - количество таргетов в данном сегменте в сверяемой выборке
        {C_TARGET_RATE_MAIN.n} - Target Rate в данному сегменте базовой выборки, 
        {C_TARGET_RATE_SECONDARY.n} - Target Rate в данному сегменте в сверяемой выборке
        {C_TARGET_POPULATION_MAIN.n} - относительный размер таргета в данном сегменте базовой выборки, 
        {C_TARGET_POPULATION_SECONDARY.n} - относительный размер таргета в данном сегменте в сверяемой выборке, 
        {C_POPULATION_STABILITY.n} - индекс стабильности размера сегмента (базовая vs сверяемая), 
        {C_TARGET_STABILITY.n} - индекс стабильности таргета сегмента (базовая vs сверяемая)
    """)


def calc_stability(
        df: DataFrame, split_var_name: str, analyze_var_name: str, split_var_value: Any = None,
        target_name: str = None, binning: Union[BinningParams, bool] = True, map_values: Dict = None,
        validate_target: bool = True, _copy: bool = True
):
    """
    1. *Бинаризуем
    2. *Маппинг
    3.

    """

    if validate_target:
        validate_binary_target(get_series_from_df(df, target_name))

    if _copy:
        df = get_sub_df(df, columns=[split_var_name, analyze_var_name, target_name])

    if binning is not False:
        if type(binning) is BinningParams:
            ser = binarize_series(
                variable=get_series_from_df(df, analyze_var_name),
                target=get_series_from_df(df, target_name),
                validate_target=False,
                cutoffs=binning.cutoffs,
                min_prc=binning.min_prc,
                rnd=binning.rnd
            )
        else:
            ser = binarize_series(
                variable=get_series_from_df(df, analyze_var_name),
                target=get_series_from_df(df, target_name),
                validate_target=False
            )
        df = set_column(df, ser, analyze_var_name)
    else:
        df = fill_missing_df(df, columns_values={analyze_var_name: MISSING})
        if map_values:
            ser = map_elements_series(get_series_from_df(df, analyze_var_name), map_values)
            df = set_column(df, ser, analyze_var_name)

    framework = get_framework_from_dataframe(df)
    func = _MAP_FRAMEWORK_calc_stability[framework]
    stability_stats = func(df, split_var_name, analyze_var_name, target_name, split_var_value)
    stability_stats = set_column(stability_stats, analyze_var_name, C_VARNAME.n)
    stability_stats = rename_columns(stability_stats,{analyze_var_name: C_VALUE.n})

    if C_TARGET_MAIN.n in get_columns(stability_stats):
        schema_out = SH_StabilityReportWithTarget.copy()
    else:
        schema_out = SH_StabilityReportNoTarget.copy()

    schema_out.replace_columns({C_SPLIT_COLUMN.n: Column(split_var_name)})
    stability_stats = schema_out(stability_stats, reorder_colums=True)
    return stability_stats


def _calc_stability_pandas(
        df: DataFrame, split_var_name: str, analyze_var_name: str,
        target_name: Union[str, None], split_var_value
) -> DataFrame:
    if split_var_value is not None:
        main_df = df[df[split_var_name] == split_var_value].reset_index(drop=True)
    else:
        main_df = df

    if target_name:
        stats_secondary = (
            df.groupby([split_var_name, analyze_var_name], as_index=False).
            agg({target_name: ['size', 'sum']})
        )
        stats_secondary.columns = [split_var_name, analyze_var_name, C_COUNT_SECONDARY.n, C_TARGET_SECONDARY.n]

        stats_main = main_df.groupby(analyze_var_name, as_index=False).agg({target_name: ['size', 'sum']})
        stats_main.columns = [analyze_var_name, C_COUNT_MAIN.n, C_TARGET_MAIN.n]

        general_stats_secondary = (
            df.groupby([split_var_name], as_index=False).
            agg({target_name: ['size', 'sum']})
        )
        general_stats_secondary.columns = [split_var_name, c_secondary_count_total, c_secondary_target_total]

    else:
        stats_secondary = df.groupby([split_var_name, analyze_var_name], as_index=False).size()
        stats_secondary.columns = [split_var_name, analyze_var_name, C_COUNT_SECONDARY.n,]

        stats_main = main_df.groupby(analyze_var_name, as_index=False).size()
        stats_main.columns = [analyze_var_name, C_COUNT_MAIN.n]

        general_stats_secondary = (
            df.groupby([split_var_name], as_index=False).size()
        )
        general_stats_secondary.columns = [split_var_name, c_secondary_count_total]

    main_total_cnt = stats_main[C_COUNT_MAIN.n].sum()
    main_target_cnt = stats_main[C_TARGET_MAIN.n].sum()

    all_groups = sorted(get_unique(get_series_from_df(df, analyze_var_name)))
    all_groups = pd.DataFrame({analyze_var_name: all_groups})

    all_splits = sorted(get_unique(get_series_from_df(df, split_var_name)))
    all_splits = pd.DataFrame({split_var_name: all_splits})

    all_groups_splits = all_splits.merge(all_groups, how='cross')
    del all_splits

    stats_main = all_groups.merge(stats_main, on=[analyze_var_name], how='left')
    stats_secondary = all_groups_splits.merge(
        stats_secondary, on=[split_var_name, analyze_var_name], how='left'
    )
    del all_groups, all_groups_splits

    stats_secondary = stats_secondary.merge(
        stats_main, on=[analyze_var_name], how='inner'
    )
    del stats_main

    stats_secondary = stats_secondary.merge(
        general_stats_secondary, how='inner', on=[split_var_name]
    )
    del general_stats_secondary

    fill_na_dict = {
        C_COUNT_MAIN.n: 0,
        C_COUNT_SECONDARY.n: 0
    }

    if target_name:
        fill_na_dict[C_TARGET_MAIN.n] = 0
        fill_na_dict[C_TARGET_SECONDARY.n] = 0

    stats_secondary = fill_missing_df(stats_secondary, fill_na_dict)

    # Необходимо у целевого убрать secondary
    if split_var_value is not None:
        main_indx = stats_secondary[split_var_name] == split_var_value
        stats_secondary.loc[main_indx, [C_COUNT_SECONDARY.n, C_TARGET_SECONDARY.n]] = None
    else:
        main_indx = None

    # Статистики общей популяции
    stats_secondary[C_POPULATION_MAIN.n] = 100 * stats_secondary[C_COUNT_MAIN.n] / main_total_cnt
    stats_secondary[C_POPULATION_SECONDARY.n] = (
            100 * stats_secondary[C_COUNT_SECONDARY.n] / stats_secondary[c_secondary_count_total]
    )
    stats_secondary[C_POPULATION_STABILITY.n] = (
        (stats_secondary[C_POPULATION_SECONDARY.n] - stats_secondary[C_POPULATION_MAIN.n]) *
        np.log(stats_secondary[C_POPULATION_SECONDARY.n] / stats_secondary[C_POPULATION_MAIN.n]).abs()
    )

    # Статистики популяции таргета
    if target_name:
        stats_secondary[C_TARGET_RATE_MAIN.n] = (
                100 * stats_secondary[C_TARGET_MAIN.n] / stats_secondary[C_COUNT_MAIN.n]
        )
        stats_secondary[C_TARGET_RATE_SECONDARY.n] = (
                100 * stats_secondary[C_TARGET_SECONDARY.n] / stats_secondary[C_COUNT_SECONDARY.n]
        )

        stats_secondary[C_TARGET_POPULATION_MAIN.n] = 100 * stats_secondary[C_TARGET_MAIN.n] / main_target_cnt
        stats_secondary[C_TARGET_POPULATION_SECONDARY.n] = (
                100 * stats_secondary[C_TARGET_SECONDARY.n] / stats_secondary[c_secondary_target_total]
        )
        stats_secondary[C_TARGET_STABILITY.n] = (
                (stats_secondary[C_TARGET_POPULATION_SECONDARY.n] - stats_secondary[C_TARGET_POPULATION_MAIN.n]) *
                np.log(
                    stats_secondary[C_TARGET_POPULATION_SECONDARY.n] / stats_secondary[C_TARGET_POPULATION_MAIN.n]
                ).abs()
        )
    return stats_secondary


def _calc_stability_polars(
        df: DataFrame, split_var_name: str, analyze_var_name: str,
        target_name: Union[str, None], split_var_value
) -> DataFrame:
    raise NotImplementedError


def _calc_stability_spark(
        df: DataFrame, split_var_name: str, analyze_var_name: str,
        target_name: Union[str, None], split_var_value
) -> DataFrame:
    raise NotImplementedError


_MAP_FRAMEWORK_calc_stability = {
    FrameWork.pandas: _calc_stability_pandas,
    FrameWork.polars: _calc_stability_polars,
    FrameWork.spark: _calc_stability_spark,
}
