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
from typing import Union, Dict, Any, List


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
    rename_columns, get_columns, map_elements_df
)
from utils.general.column import Column
from utils.general.schema import SchemaDF
from utils.general.types import DataFrame, FrameWork, get_framework_from_dataframe

from preprocessing.binning import binarize_series, BinningParams

from ._pandas import calc_stability as _calc_stability_pandas
from ._polars import calc_stability as _calc_stability_polars

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
        df: DataFrame, split_var_name: str, analyze_vars: List[str], split_var_value: Any = None,
        target_name: str = None, binning: Union[Dict[str, Union[BinningParams, bool]], bool, BinningParams] = True,
        map_values: Dict = None, validate_target: bool = True, _copy: bool = True
):
    """
    1. *Бинаризуем
    2. *Маппинг
    3.

    """

    if validate_target:
        validate_binary_target(get_series_from_df(df, target_name))

    if _copy:
        df = get_sub_df(df, columns=[split_var_name, target_name] + analyze_vars)

    if type(binning) is dict:
        replace_dict = {var_name: MISSING for var_name in analyze_vars if var_name not in binning}
        df = fill_missing_df(df, columns_values=replace_dict)

    if map_values:
        df = map_elements_df(df, map_values)

    if binning is True:
        binning = {col: True for col in analyze_vars}
    elif binning is False:
        binning = {}
    elif type(binning) is BinningParams:
        binning = {col: binning for col in analyze_vars}

    for var_name in binning:
        if binning[var_name] is True:
            ser = binarize_series(
                variable=get_series_from_df(df, var_name),
                target=get_series_from_df(df, target_name),
                validate_target=False
            )
            df = set_column(df, ser, var_name)

        else:
            ser = binarize_series(
                variable=get_series_from_df(df, var_name),
                target=get_series_from_df(df, target_name),
                validate_target=False,
                cutoffs=binning[var_name].cutoffs,
                min_prc=binning[var_name].min_prc,
                rnd=binning[var_name].rnd
            )
            df = set_column(df, ser, var_name)

    framework = get_framework_from_dataframe(df)
    func = _MAP_FRAMEWORK_calc_stability[framework]
    stability_stats = func(df, split_var_name, analyze_vars, target_name, split_var_value, _sort=True)

    rename_dict = {}
    names_cols = []
    values_cols = []
    for i, var_name in enumerate(analyze_vars):
        C_VARNAME_I = C_VARNAME.child(f'Var{i+1}_Name')
        C_VALUE_I = C_VALUE.child(f'Var{i+1}_Value')
        stability_stats = set_column(stability_stats, var_name, C_VARNAME_I.n)
        rename_dict[var_name] = C_VALUE_I.n

        names_cols.append(C_VARNAME_I)
        values_cols.append(C_VALUE_I)

    stability_stats = rename_columns(stability_stats, rename_dict)

    if C_TARGET_MAIN.n in get_columns(stability_stats):
        schema_out = SH_StabilityReportWithTarget.copy()
    else:
        schema_out = SH_StabilityReportNoTarget.copy()

    schema_out.delete_columns([C_VARNAME, C_VALUE])
    schema_out.add_columns({col: True for col in names_cols + values_cols}, pos=1)
    schema_out.replace_columns({C_SPLIT_COLUMN.n: Column(split_var_name)})

    stability_stats = schema_out(stability_stats, reorder_colums=True)
    return stability_stats


_MAP_FRAMEWORK_calc_stability = {
    FrameWork.pandas: _calc_stability_pandas,
    FrameWork.polars: _calc_stability_polars,
    # FrameWork.spark: _calc_stability_spark,
}
