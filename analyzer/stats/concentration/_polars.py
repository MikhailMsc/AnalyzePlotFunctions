import re
from typing import List, Tuple, Union, Dict

from analyzer.utils.domain.columns import (
    C_COUNT, C_TARGET, C_POPULATION, C_TARGET_POPULATION, C_TARGET_RATE, C_GROUP_IV,
    C_SEGMENT_ID, C_PARENT_MIN, C_PARENT_MAX, C_PARENT_MIN_TR, C_PARENT_MAX_TR
)
from analyzer.utils.framework_depends import get_columns
from analyzer.utils.general.types import DataFrame


regex_varname = re.compile('Var[0-9]+_Name')
regex_varvalue = re.compile('Var[0-9]+_Value')


def calc_concentration(
        df: DataFrame, analyze_vars: List[str], target_name: str, start_id: int, target_cnt: int
) -> Tuple[DataFrame, int]:
    import polars as pl

    total_cnt = df.shape[0]
    not_target_cnt = total_cnt - target_cnt

    stats = df.group_by(analyze_vars).agg(
        pl.col(target_name).count().alias(C_COUNT.n),
        pl.col(target_name).sum().alias(C_TARGET.n)
    )

    target_population = stats[C_TARGET.n] / target_cnt

    not_target = stats[C_COUNT.n] - stats[C_TARGET.n]
    not_target_population = not_target / not_target_cnt

    stats = stats.with_columns(
        pl.int_range(start_id, start_id + pl.len(), dtype=pl.UInt32).alias(C_SEGMENT_ID.n),
        (100 * pl.col(C_COUNT.n) / total_cnt).round(2).alias(C_POPULATION.n),
        (100 * target_population).round(2).alias(C_TARGET_POPULATION.n),
        (100 * pl.col(C_TARGET.n) / pl.col(C_COUNT.n)).alias(C_TARGET_RATE.n),
        (
            100 * (target_population - not_target_population).abs() *
            (
                pl.when(target_population == 0).then(1 / (target_cnt or not_target_cnt)).otherwise(target_population) /
                pl.when(not_target_population == 0).then(1 / (not_target_cnt or target_cnt)).otherwise(not_target_population)
            ).log()
        ).alias(C_GROUP_IV.n),
    )
    return stats, stats[C_SEGMENT_ID.n][-1]


def merge_nearest_segments_pl(report, map_func):
    import polars as pl
    report = report.with_columns(
        pl.col(C_SEGMENT_ID.n).map_elements(map_func, return_dtype=pl.List(pl.List(pl.Int64))).
        alias('min_max_segments')
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


def extract_dict_segments_polars(df: DataFrame, vars_order: Dict):
    import polars as pl
    var_name_cols = [col for col in df.columns if regex_varname.match(col)]
    var_value_cols = [col for col in df.columns if regex_varvalue.match(col)]
    var_cols = zip(var_name_cols, var_value_cols)
    segments = df.select(pl.concat_list([col for combo in var_cols for col in combo]).alias('segment'))

    def make_segment(arr):
        return [(arr[i], arr[i+1]) for i in range(0, len(arr), 2) if arr[i+1] is not None]

    segments = segments.with_columns(pl.col('segment').map_elements(
        make_segment, return_dtype=pl.List(pl.List(pl.Int64))
    ).alias('segment'))
    segment_info = [tuple([tuple(combo) for combo in seg]) for seg in segments['segment'].to_list()]
    segment_id = df[C_SEGMENT_ID.n].to_list()
    target_rate = df[C_TARGET_RATE.n].to_list()
    segments = {
        sid: {
            'segment': info,
            'target_rate': tgr
        }
        for sid, tgr, info in zip(segment_id, target_rate, segment_info)
    }

    return segments
