import re
from typing import List, Dict, Tuple, Union

from utils.domain.columns import C_COUNT, C_TARGET, C_POPULATION, C_TARGET_POPULATION, C_TARGET_RATE, C_WOE, C_GROUP_IV, \
    C_SEGMENT_ID, C_PARENT_MIN, C_PARENT_MAX, C_PARENT_MIN_TR, C_PARENT_MAX_TR
from utils.domain.const import EPSILON
from utils.framework_depends import get_columns
from utils.general.types import DataFrame


# info_msg = f"""
#     Var[i]_Name - имя i-ой переменной участвующей в сегменте,
#     Var[i]_Value - значение i-ой переменной участвующей в сегменте,
#     {C_COUNT.n} - количество наблюдений в данном сегменте,
#     {C_POPULATION.n} - относительный размер сегмента,
#     {C_TARGET.n} - количество таргетов в данном сегменте,
#     {C_TARGET_RATE.n} - Target Rate в данном сегменте,
#     {C_TARGET_POPULATION.n} - относительный размер таргета в данном сегменте,
#     {C_GROUP_IV.n} - information value данного сегмента,
#
#     {C_PARENT_SEGMENT.n} - родительский сегмент (ближайший по target_rate),
#     {C_PARENT_COUNT.n} - количество наблюдений в родительском сегменте,
#     {C_POPULATION_PARENT.n} - относительный размер родительского сегмента,
#     {C_PARENT_TARGET.n} - количество таргетов в родительском сегменте,
#     {C_PARENT_TARGET_RATE.n} - Target Rate в родительском сегменте,
#     {C_PARENT_TARGET_POPULATION.n} - относительный размер таргета в родительском сегменте,
#     {C_PARENT_GROUP_IV.n} - information value родительского сегмента
#
#
#     Information Value (IV):
#     - Чем больше значение ПО МОДУЛЮ = тем больше отклонение от target rate oт среднего по выборке
#     - IV > 0 = target rate выше среднего по выборке
#     - IV < 0 = target rate ниже среднего по выборке
# """

regex_varname = re.compile('Var[0-9]+_Name')
regex_varvalue = re.compile('Var[0-9]+_Value')


def calc_concentration(
        df: DataFrame, analyze_vars: List[str], target_name: str, vars_order: Dict[str, int], start_id: int
) -> Tuple[DataFrame, Dict]:
    import polars as pl

    total_cnt = df.shape[0]
    target_cnt = df[target_name].sum()
    not_target_cnt = total_cnt - target_cnt

    stats = df.group_by(analyze_vars).agg(
        pl.col(target_name).count().alias(C_COUNT.n),
        pl.col(target_name).sum().alias(C_TARGET.n)
    )

    target_population = stats[C_TARGET.n] / target_cnt

    not_target = stats[C_COUNT.n] - stats[C_TARGET.n]
    not_target_population = not_target / not_target_cnt

    stats = stats.with_columns(
        pl.Series(list(range(start_id, start_id + stats.shape[0]))).alias(C_SEGMENT_ID.n),
        (100 * pl.col(C_COUNT.n) / total_cnt).round(2).alias(C_POPULATION.n),
        (100 * target_population).round(2).alias(C_TARGET_POPULATION.n),
        (100 * pl.col(C_TARGET.n) / pl.col(C_COUNT.n)).alias(C_TARGET_RATE.n),

    )

    stats = stats.with_columns(
        (
            target_population /
            pl.when(not_target_population == 0).then(EPSILON).otherwise(not_target_population)
        ).log().alias(C_WOE.n)
    )
    stats = stats.with_columns(
        (100 * (target_population - not_target_population).abs() * pl.col(C_WOE.n)).alias(C_GROUP_IV.n),
    )

    segmets = {
        stats[C_SEGMENT_ID.n][row_id]: {
            'segment': tuple([(vars_order[var], str(val)) for var, val in zip(analyze_vars, arr)]),
            'target_rate': stats[C_TARGET_RATE.n][row_id]

        }
        for row_id, arr in enumerate(stats[analyze_vars].to_numpy())
    }
    return stats, segmets


def merge_nearest_segments_pl(report, map_func):
    import polars as pl
    report = report.with_columns(
        pl.col(C_SEGMENT_ID.n).map_elements(map_func).alias('min_max_segments')
    )
    report = report.with_columns(
        pl.col('min_max_segments').list.get(0).list.get(0).alias('min_tg_segment_id'),
        pl.col('min_max_segments').list.get(0).list.get(1).alias(C_PARENT_MIN.n),
        pl.col('min_max_segments').list.get(1).list.get(0).alias('max_tg_segment_id'),
        pl.col('min_max_segments').list.get(1).list.get(1).alias(C_PARENT_MAX.n),
    )

    report_copy = report[[C_SEGMENT_ID.n, C_TARGET_RATE.n]]
    report_copy = report_copy.rename({C_SEGMENT_ID.n: 'min_tg_segment_id', C_TARGET_RATE.n: C_PARENT_MIN_TR.n})
    report = report.join(report_copy, on=['min_tg_segment_id'], how='left')

    report_copy = report_copy.rename({'min_tg_segment_id': 'max_tg_segment_id', C_PARENT_MIN_TR.n: C_PARENT_MAX_TR.n})
    report = report.join(report_copy, on=['max_tg_segment_id'], how='left')
    return report


def make_reverse_mapping_pl(report: DataFrame, reverse_map_vars: dict, vars_revers_order: dict) -> DataFrame:
    import polars as pl

    varname_i_cols = [col for col in get_columns(report) if regex_varname.match(col)]
    parent_cols = [C_PARENT_MAX.n, C_PARENT_MIN.n]
    cnt_vars = len(varname_i_cols)

    report = report.with_columns(*[
        pl.col(col_nm).map_elements(lambda x: vars_revers_order.get(x, None), return_dtype=pl.String).alias(col_nm)
        for col_nm in varname_i_cols
    ])

    def map_col(x):
        if x is None:
            return None
        elif x not in vars_revers_order:
            return None
        else:
            return '- ' + vars_revers_order[x]

    report = report.with_columns(*[
        pl.col(col_nm).map_elements(map_col, return_dtype=pl.String).alias(col_nm)
        for col_nm in parent_cols
    ])

    def f_mapping(row):
        var_name = row[var_name_column]
        if var_name is None:
            return None

        var_value = row[var_value_column]
        return str(reverse_map_vars[var_name][var_value])

    for i in range(cnt_vars):
        var_name_column = f'Var{i+1}_Name'
        var_value_column = f'Var{i + 1}_Value'
        report = report.with_columns(
            pl.struct([var_name_column, var_value_column]).map_elements(f_mapping, return_dtype=pl.String).
            alias(var_value_column)
        )
    return report


def filter_by_population_pl(report: DataFrame, pop_more: Union[float, None]):
    import polars as pl

    if pop_more is not None:
        report = report.filter(pl.col(C_POPULATION.n) >= pop_more)
    return report


def filter_by_tgrate_pl(report: DataFrame, tr_less, tr_more):
    import polars as pl

    if tr_less is not None and tr_more is not None:
        report = report.filter((pl.col(C_TARGET_RATE.n) <= tr_less) | (pl.col(C_TARGET_RATE.n) >= tr_more))
    elif tr_less is not None:
        report = report.filter(pl.col(C_TARGET_RATE.n) <= tr_less)
    elif tr_more is not None:
        report = report.filter(pl.col(C_TARGET_RATE.n) >= tr_more)
    return report
