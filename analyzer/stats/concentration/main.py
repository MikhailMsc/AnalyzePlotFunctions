import itertools
from typing import List, Dict
import math

from tqdm import tqdm

from analyzer import logger
from analyzer.preprocessing import BinningParamsMultiVars, MapDictMultiVars, preprocess_df
from analyzer.utils.domain.columns import (
    C_COUNT, C_POPULATION, C_TARGET, C_TARGET_RATE, C_TARGET_POPULATION, C_GROUP_IV,
    C_VARNAME, C_VALUE, C_PARENT_MIN, C_PARENT_MAX, C_PARENT_MIN_TR, C_PARENT_MAX_TR, C_SEGMENT_ID
)
from analyzer.utils.framework_depends import (
    get_columns, get_sub_df, encode_df, optimize_df_int_types, set_column,
    rename_columns, convert_df_to_polars, concat_df, convert_df_to_pandas, get_nunique
)
from analyzer.utils.general.schema import SchemaDF
from analyzer.utils.general.types import DataFrame, get_framework_from_dataframe, FrameWork

from ._polars import (
    calc_concentration as calc_concentration_polars, merge_nearest_segments_pl,
    make_reverse_mapping_pl, filter_by_population_pl, filter_by_tgrate_pl, extract_dict_segments_polars
)

info_msg = f"""
    Var[i]_Name - имя i-ой переменной участвующей в сегменте,
    Var[i]_Value - значение i-ой переменной участвующей в сегменте,
    {C_COUNT.n} - количество наблюдений в данном сегменте,
    {C_POPULATION.n} - относительный размер сегмента,
    {C_TARGET.n} - количество таргетов в данном сегменте,
    {C_TARGET_RATE.n} - Target Rate в данном сегменте,
    {C_TARGET_POPULATION.n} - относительный размер таргета в данном сегменте,
    {C_GROUP_IV.n} - information value данного сегмента,

    {C_PARENT_MIN.n} - родительский сегмент с минимальным target_rate,
    {C_PARENT_MIN_TR.n} - минимальный target_rate родительского сегмента,
    {C_PARENT_MAX.n} - родительский сегмент с максимальным target_rate,
    {C_PARENT_MAX_TR.n} - максимальный target_rate родительского сегмента

    Information Value (IV):
    - Чем больше значение ПО МОДУЛЮ = тем больше отклонение target rate oт среднего по выборке (сильнее разделяющая способность)
    - IV > 0 = target rate выше среднего по выборке
    - IV < 0 = target rate ниже среднего по выборке
"""

SH_ConcentrationReport = SchemaDF(
    columns=[
        C_VARNAME, C_VALUE, C_COUNT, C_POPULATION, C_TARGET, C_TARGET_RATE,
        C_TARGET_POPULATION, C_GROUP_IV, C_PARENT_MIN, C_PARENT_MIN_TR, C_PARENT_MAX, C_PARENT_MAX_TR
    ],
    key=[C_VARNAME, C_VALUE]
)


def calc_concentration_report(
        df: DataFrame, target_name: str, combo_min: int = 1, combo_max: int = 1,
        analyze_vars: List[str] = None, ignore_vars: List[str] = None,
        map_values: MapDictMultiVars = None,
        binning: BinningParamsMultiVars = True,
        pop_more: float = None,
        tr_less: float = None,
        tr_more: float = None,
        sort_by_iv: bool = False,
        _tqdm: bool = True,
        _validate_target: bool = True,
        _bin_by_target: bool = True,
        _logging: bool = True,
        _drop_single_vals: bool = True
) -> DataFrame:
    """
    Отчет по концентрации таргета. Поиск аномальных сегментов.

    Args:
        df:                     Исследуемый датафрейм
        target_name:            Название таргета
        combo_min:              Минимальное количество переменных, участвующих в комбинировании
        combo_max:              Максимальное количество переменных, участвующих в комбинировании
        analyze_vars:           Список анализируемых переменных, опционально
        ignore_vars:            Список игнорируемых переменных, опционально
        map_values:             Словарь для замены значений переменных (словарь, ключ = название переменной,
                                значение = словарь старое-новое значение)
        binning:                Параметры для биннинга
        pop_more:               Удалить сегменты, менее заданного процента популяции
        tr_less:                Оставить сегменты, TargetRate которых менее заданного процента
        tr_more:                Оставить сегменты, TargetRate которых выше заданного процента
        sort_by_iv:             Отсортировать по IV
        _tqdm:                  Отображать прогрессбар
        _validate_target:       Валидация таргета
        _bin_by_target:         Бинаризация с использованием таргета
        _logging:               Логгировать сообщения
        _drop_single_vals:      Удалить из анализа переменные имеющие одно значение/категорию

    Returns:
        DataFrame:
            Var[i]_Name:                имя i-ой переменной участвующей в сегменте,
            Var[i]_Value:               значение i-ой переменной участвующей в сегменте,
            COUNT:                      количество наблюдений в данном сегменте,
            POPULATION:                 относительный размер сегмента,
            TARGET:                     количество таргетов в данном сегменте,
            TARGET_RATE:                Target Rate в данном сегменте,
            TARGET_POPULATION:          относительный размер таргета в данном сегменте,
            GROUP_IV:                   information value данного сегмента,
            PARENT_MIN:                 родительский сегмент с минимальным target_rate,
            PARENT_MIN_TargetRate:      минимальный target_rate родительского сегмента,
            PARENT_MAX:                 родительский сегмент с максимальным target_rate,
            PARENT_MAX_TargetRate:      максимальный target_rate родительского сегмента

    Примечание. Information Value (IV):
        - Чем больше значение ПО МОДУЛЮ = тем больше отклонение target rate oт среднего по выборке (сильнее разделяющая способность)
        - IV > 0 = target rate выше среднего по выборке
        - IV < 0 = target rate ниже среднего по выборке
    """

    if not analyze_vars:
        analyze_vars = set(get_columns(df))
        if ignore_vars:
            analyze_vars = analyze_vars - set(ignore_vars)

        analyze_vars.remove(target_name)
        analyze_vars = list(analyze_vars)

    assert target_name not in analyze_vars
    analyze_vars = sorted(analyze_vars)

    select_cols = analyze_vars + [target_name, ]
    df = get_sub_df(df, select_cols)

    framework = get_framework_from_dataframe(df)
    if framework is FrameWork.pandas:
        df = convert_df_to_polars(df)
        framework = FrameWork.polars

    df = preprocess_df(
        df, analyze_vars, ignore_vars=None, target_name=target_name,
        map_values=map_values, binning=binning, drop_not_processed=False,
        _validate_target=_validate_target, _bin_by_target=_bin_by_target, _copy=False
    )
    df, reverse_map_vars, min_max_values = encode_df(df, analyze_vars, bad_vars_behavior='drop')
    analyze_vars = [col for col in analyze_vars if col in get_columns(df)]
    df = optimize_df_int_types(df, min_max_values)

    if _drop_single_vals:
        var_single_value = [var for var, cnt in get_nunique(df, columns=analyze_vars).items() if cnt == 1]
        if len(var_single_value) > 0:
            msg = f'Некоторые переменные ({len(var_single_value)}) имеют одно значение, они будут исключены из анализа:\n\t'
            msg = msg + '\n\t'.join(var_single_value)
            logger.log_debug(msg)
            var_single_value = set(var_single_value)
            analyze_vars = [var for var in analyze_vars if var not in var_single_value]

    vars_order = {var: i for i, var in enumerate(analyze_vars)}
    vars_revers_order = {i: var for var, i in vars_order.items()}

    full_report = []
    start_id_segment = 1
    cnt_target = df[target_name].sum()

    for cnt in range(combo_min, combo_max + 1):
        cnt_combos = math.comb(len(analyze_vars), cnt)
        iter_obj = itertools.combinations(analyze_vars, cnt)
        if _tqdm:
            bar_format = (f"{{l_bar}}{{bar}}| Комбинации из {cnt} переменных, {{n_fmt}}/{cnt_combos} "
                          f"[{{elapsed}}<{{remaining}}, {{rate_fmt}}]")
            iter_obj = tqdm(iter_obj, total=cnt_combos, bar_format=bar_format)

        for vars_combo in iter_obj:
            combo_report, max_id_segment = _calc_concentration_combo(
                df, vars_combo, target_name, vars_order, start_id_segment, cnt_target)
            start_id_segment = max_id_segment + 1
            full_report.append(combo_report)

    full_report = concat_df(full_report, vertical=True)
    dict_segments = _extract_dict_segments(full_report, vars_order)

    reverse_dict_segments = {
        seg_info['segment']: {'id': seg_id, 'target_rate': seg_info['target_rate']}
        for seg_id, seg_info in dict_segments.items()
    }

    def get_corner_segments(id_segment):
        current_segment = dict_segments[id_segment]
        current_cnt_vars = len(current_segment['segment'])
        current_target_rate = current_segment['target_rate']
        current_segment = current_segment['segment']

        if current_cnt_vars == 1:
            return (-1, -1), (-1, -1)

        nearest_segments = []

        for parent_segment in itertools.combinations(current_segment, current_cnt_vars - 1):
            parent_segment = tuple(sorted(parent_segment, key=lambda x: x[0]))
            parent_segment_info = reverse_dict_segments[parent_segment]
            delta = set(current_segment) - set(parent_segment if len(parent_segment) != 1 else (parent_segment[0], ))
            assert len(delta) == 1
            delta = delta.pop()[0]
            nearest_segments.append(
                (parent_segment_info['id'], delta, current_target_rate - parent_segment_info['target_rate'])
            )

        min_tgrate_segment = max(nearest_segments, key=lambda x: x[2])[:2]
        max_tgrate_segment = min(nearest_segments, key=lambda x: x[2])[:2]

        return min_tgrate_segment, max_tgrate_segment

    if framework is FrameWork.polars:
        full_report = filter_by_population_pl(full_report, pop_more)
        full_report = merge_nearest_segments_pl(full_report, get_corner_segments)
        full_report = make_reverse_mapping_pl(full_report, reverse_map_vars, vars_revers_order)
        full_report = filter_by_tgrate_pl(full_report, tr_less, tr_more)
    else:
        raise NotImplementedError

    schema_out = SH_ConcentrationReport.copy()
    schema_out.delete_columns([C_VARNAME, C_VALUE])
    schema_out.add_columns({
        col.child(f'Var{i}_{col_name}'): True
        for i in range(combo_min, combo_max + 1)
        for col_name, col in zip(['Name', 'Value'], [C_VARNAME, C_VALUE])
    }, pos=0)
    full_report = schema_out(full_report, reorder_colums=True)
    full_report = convert_df_to_pandas(full_report)
    if sort_by_iv:
        full_report = full_report.sort_values(C_GROUP_IV.n, ascending=False)
        full_report.reset_index(drop=True, inplace=True)

    if _logging:
        logger.log_info(info_msg)
    return full_report


def _calc_concentration_combo(
        df: DataFrame, analyze_vars: List[str], target_name: str,
        vars_order: Dict[str, int], start_id: int, cnt_target: int
):
    analyze_vars = sorted(analyze_vars)
    framework = get_framework_from_dataframe(df)
    func = _MAP_FRAMEWORK_calc_concentration[framework]

    concentr_stats, max_semgent_id = func(df, analyze_vars, target_name, start_id, cnt_target)
    rename_dict = {}
    names_cols = []
    values_cols = []
    for i, var_name in enumerate(analyze_vars):
        C_VARNAME_I = C_VARNAME.child(f'Var{i+1}_Name')
        C_VALUE_I = C_VALUE.child(f'Var{i+1}_Value')
        concentr_stats = set_column(concentr_stats, vars_order[var_name], C_VARNAME_I.n)
        rename_dict[var_name] = C_VALUE_I.n

        names_cols.append(C_VARNAME_I)
        values_cols.append(C_VALUE_I)

    concentr_stats = rename_columns(concentr_stats, rename_dict)
    return concentr_stats, max_semgent_id


_MAP_FRAMEWORK_calc_concentration = {
    FrameWork.polars: calc_concentration_polars
}


def _extract_dict_segments(df: DataFrame, vars_order: Dict):
    framework = get_framework_from_dataframe(df)
    func = _MAP_FRAMEWORK_extract_dict_segments[framework]
    return func(df, vars_order)


_MAP_FRAMEWORK_extract_dict_segments = {
    FrameWork.polars: extract_dict_segments_polars
}
