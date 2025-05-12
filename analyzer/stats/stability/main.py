import itertools
import re
from typing import Union, Any, List

from analyzer.preprocessing import BinningParamsMultiVars, preprocess_df, MapDictMultiVars
from analyzer.utils.domain.columns import (
    C_COUNT_MAIN, C_COUNT_SECONDARY, C_POPULATION_MAIN, C_POPULATION_SECONDARY,
    C_TARGET_MAIN, C_TARGET_SECONDARY, C_TARGET_RATE_MAIN, C_TARGET_RATE_SECONDARY,
    C_TARGET_POPULATION_MAIN, C_TARGET_POPULATION_SECONDARY, C_POPULATION_STABILITY,
    C_TARGET_STABILITY, C_VARNAME, C_VALUE, C_SPLIT_COLUMN
)
from analyzer.utils.framework_depends import (
    set_column, rename_columns, get_columns, convert_df_to_pandas, get_sub_df, encode_df, optimize_df_int_types,
    concat_df, convert_df_to_polars
)
from analyzer.utils.general.column import Column
from analyzer.utils.general.schema import SchemaDF
from analyzer.utils.general.types import DataFrame, FrameWork, get_framework_from_dataframe

from ._pandas import calc_stability as _calc_stability_pandas, make_reverse_mapping_pandas, filter_small_segments_pandas
from ._polars import calc_stability as _calc_stability_polars, make_reverse_mapping_polars, filter_small_segments_polars

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

regex_varname = re.compile('Var[0-9]+_Name')
regex_varvalue = re.compile('Var[0-9]+_Value')

info_msg = f"""
    {C_SPLIT_COLUMN.n} - поле, которое разбивает данные на подвыборки для сверки их характеристик между собой,
    Var[i]_Name - имя i-ой переменной участвующей в сегменте, 
    Var[i]_Value - значение i-ой переменной участвующей в сегменте, 
    {C_COUNT_MAIN.n} - количество наблюдений в данном сегменте базовой выборки (относительно которой будем сверять),
    {C_COUNT_SECONDARY.n} - количество наблюдений в данном сегменте в сверяемой выборке (определяется по {C_SPLIT_COLUMN.n}),
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
    
    Показатели стабильности:
    - Чем больше значение ПО МОДУЛЮ = тем больше отклонение от базового значения
    - Стабильность > 0 = целевой показатель увеличился, относительно базового значения.
    - Стабильность < 0 = целевой показатель уменьшился, относительно базового значения.
"""


def calc_stability_report(
        df: DataFrame, split_var_name: str, combo_min: int = 1, combo_max: int = 1,
        analyze_vars: List[str] = None, ignore_vars: List[str] = None,
        split_var_value: Any = None, target_name: str = None,
        binning: BinningParamsMultiVars = True,
        map_values: MapDictMultiVars = None,
        min_part_or_cnt: Union[float, int] = None,
        min_abs_popstab: float = None,
        min_abs_tgstab: float = None

) -> DataFrame:

    if not analyze_vars:
        analyze_vars = set(get_columns(df))
        analyze_vars.remove(split_var_name)
        if ignore_vars:
            analyze_vars = analyze_vars - set(ignore_vars)

        if target_name:
            analyze_vars.remove(target_name)
        analyze_vars = list(analyze_vars)

    select_cols = analyze_vars[:] + [split_var_name, ]
    if target_name:
        select_cols.append(target_name)

    df = get_sub_df(df, select_cols)
    if get_framework_from_dataframe(df) is FrameWork.pandas:
        df = convert_df_to_polars(df)

    df = preprocess_df(
        df, analyze_vars, None, target_name, binning,
        map_values, _validate_target=True, drop_not_processed=False
    )
    df, reverse_map_vars, min_max_values = encode_df(df, analyze_vars)
    df = optimize_df_int_types(df, min_max_values)

    full_report = []
    for cnt in range(combo_min, combo_max + 1):
        for vars_combo in itertools.combinations(analyze_vars, cnt):
            combo_report = _calc_stability_combo(
                df, split_var_name, vars_combo, split_var_value, target_name
            )
            full_report.append(combo_report)

    full_report = concat_df(full_report, vertical=True)
    full_report = _filter_small_segments(full_report, min_part_or_cnt, min_abs_popstab, min_abs_tgstab)
    full_report = _make_reverse_mapping(full_report, reverse_map_vars)

    if target_name:
        schema_out = SH_StabilityReportWithTarget.copy()
    else:
        schema_out = SH_StabilityReportNoTarget.copy()

    schema_out.delete_columns([C_VARNAME, C_VALUE])
    schema_out.add_columns({
        col.child(f'Var{i}_{col_name}'): True
        for i in range(combo_min, combo_max + 1)
        for col_name, col in zip(['Name', 'Value'], [C_VARNAME, C_VALUE])
    }, pos=1)
    schema_out.replace_columns({C_SPLIT_COLUMN.n: Column(split_var_name)})
    full_report = schema_out(full_report, reorder_colums=True)
    full_report = convert_df_to_pandas(full_report)
    print(info_msg)
    return full_report


def _calc_stability_combo(
        df: DataFrame, split_var_name: str, analyze_vars: List[str],
        split_var_value: Any = None, target_name: str = None
) -> Union[SH_StabilityReportWithTarget.t, SH_StabilityReportNoTarget.t]:

    analyze_vars = sorted(analyze_vars)

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
    return stability_stats


_MAP_FRAMEWORK_calc_stability = {
    FrameWork.pandas: _calc_stability_pandas,
    FrameWork.polars: _calc_stability_polars,
    # FrameWork.spark: _calc_stability_spark,
}


def _filter_small_segments(
        report: DataFrame, min_part_or_cnt: Union[float, int] = None, min_abs_popstab: float = None,
        min_abs_tgstab: float = None
) -> DataFrame:
    filter_dict = dict()

    if min_part_or_cnt is not None:
        assert min_part_or_cnt >= 0
        if type(min_part_or_cnt) is float:
            assert min_part_or_cnt < 1.0
            filter_dict[C_POPULATION_SECONDARY.n] = 100 * min_part_or_cnt

        else:
            filter_dict[C_COUNT_SECONDARY.n] = min_part_or_cnt

    if min_abs_popstab is not None:
        assert min_abs_popstab > 0
        filter_dict[C_POPULATION_STABILITY.n] = min_abs_popstab

    if min_abs_tgstab is not None and C_TARGET_STABILITY.n in get_columns(report):
        assert min_abs_tgstab > 0
        filter_dict[C_TARGET_STABILITY.n] = min_abs_tgstab

    if not filter_dict:
        return report

    unique_cols = [col for col in get_columns(report) if regex_varname.match(col) or regex_varvalue.match(col)]
    framework = get_framework_from_dataframe(report)
    func = _MAP_FRAMEWORK_filter_small_segments[framework]
    return func(report, filter_dict, unique_cols)


_MAP_FRAMEWORK_filter_small_segments = {
    FrameWork.pandas: filter_small_segments_pandas,
    FrameWork.polars: filter_small_segments_polars,
    # FrameWork.spark: _calc_stability_spark,
}


def _make_reverse_mapping(report: DataFrame, reverse_map_vars: dict) -> DataFrame:
    cnt_vars = len([col for col in get_columns(report) if regex_varname.match(col)])
    framework = get_framework_from_dataframe(report)
    func = _MAP_FRAMEWORK_make_reverse_mapping[framework]

    for i in range(cnt_vars):
        var_name_column = f'Var{i+1}_Name'
        var_value_column = f'Var{i + 1}_Value'
        report = func(report, var_name_column, var_value_column, reverse_map_vars)
    return report


_MAP_FRAMEWORK_make_reverse_mapping = {
    FrameWork.pandas: make_reverse_mapping_pandas,
    FrameWork.polars: make_reverse_mapping_polars,
    # FrameWork.spark: _calc_stability_spark,
}
