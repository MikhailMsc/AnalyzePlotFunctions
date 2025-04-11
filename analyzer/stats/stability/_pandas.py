from functools import reduce
from typing import Union, List

import numpy as np

from analyzer.utils.domain.columns import (
    C_COUNT_SECONDARY, C_TARGET_SECONDARY, C_TARGET_MAIN, C_COUNT_MAIN, C_POPULATION_MAIN,
    C_POPULATION_SECONDARY, C_POPULATION_STABILITY, C_TARGET_RATE_MAIN, C_TARGET_RATE_SECONDARY,
    C_TARGET_POPULATION_MAIN, C_TARGET_POPULATION_SECONDARY, C_TARGET_STABILITY
)
from analyzer.utils.domain.const import EPSILON
from analyzer.utils.framework_depends import fill_missing_df
from analyzer.utils.general.types import DataFrame

from ._const import c_secondary_count_total, c_secondary_target_total


def calc_stability(
        df: DataFrame, split_var_name: str, analyze_vars: List[str],
        target_name: Union[str, None], split_var_value, _sort: bool = True
) -> DataFrame:
    if split_var_value is not None:
        main_df = df[df[split_var_name] == split_var_value].reset_index(drop=True)
    else:
        main_df = df

    grp_cols = [split_var_name, ] + analyze_vars
    if target_name:
        stats_secondary = (
            df.groupby(grp_cols, as_index=False).
            agg({target_name: ['size', 'sum']})
        )
        stats_secondary.columns = grp_cols + [C_COUNT_SECONDARY.n, C_TARGET_SECONDARY.n]

        stats_main = main_df.groupby(analyze_vars, as_index=False).agg({target_name: ['size', 'sum']})
        stats_main.columns = analyze_vars + [C_COUNT_MAIN.n, C_TARGET_MAIN.n]

        general_stats_secondary = (
            df.groupby([split_var_name], as_index=False).
            agg({target_name: ['size', 'sum']})
        )
        general_stats_secondary.columns = [split_var_name, c_secondary_count_total, c_secondary_target_total]

    else:

        stats_secondary = df.groupby(grp_cols, as_index=False).size()
        stats_secondary.columns = grp_cols + [C_COUNT_SECONDARY.n,]

        stats_main = main_df.groupby(analyze_vars, as_index=False).size()
        stats_main.columns = analyze_vars + [C_COUNT_MAIN.n]

        general_stats_secondary = (
            df.groupby([split_var_name], as_index=False).size()
        )
        general_stats_secondary.columns = [split_var_name, c_secondary_count_total]

    main_total_cnt = stats_main[C_COUNT_MAIN.n].sum()
    main_target_cnt = stats_main[C_TARGET_MAIN.n].sum()

    all_groups = df[analyze_vars].drop_duplicates(ignore_index=True)
    all_splits = df[[split_var_name]].drop_duplicates(ignore_index=True)

    all_groups_splits = all_splits.merge(all_groups, how='cross')
    del all_splits

    stats_main = all_groups.merge(stats_main, on=analyze_vars, how='left')
    stats_secondary = all_groups_splits.merge(
        stats_secondary, on=grp_cols, how='left'
    )
    del all_groups, all_groups_splits

    stats_secondary = stats_secondary.merge(
        stats_main, on=analyze_vars, how='inner'
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
    # if split_var_value is not None:
    #     main_indx = stats_secondary[split_var_name] == split_var_value
    #     stats_secondary.loc[main_indx, [C_COUNT_SECONDARY.n, C_TARGET_SECONDARY.n]] = None

    # Статистики общей популяции
    stats_secondary[C_POPULATION_MAIN.n] = 100 * stats_secondary[C_COUNT_MAIN.n] / main_total_cnt
    stats_secondary[C_POPULATION_SECONDARY.n] = (
            100 * stats_secondary[C_COUNT_SECONDARY.n] / stats_secondary[c_secondary_count_total]
    )
    stats_secondary[C_POPULATION_STABILITY.n] = (
        (stats_secondary[C_POPULATION_SECONDARY.n] - stats_secondary[C_POPULATION_MAIN.n]) *
        np.log(
            stats_secondary[C_POPULATION_SECONDARY.n].replace(0, EPSILON) /
            stats_secondary[C_POPULATION_MAIN.n].replace(0, EPSILON)
        ).abs()
    )

    # Статистики популяции таргета
    if target_name:
        stats_secondary[C_TARGET_RATE_MAIN.n] = (
                100 * stats_secondary[C_TARGET_MAIN.n] / stats_secondary[C_COUNT_MAIN.n].replace(0, EPSILON)
        )
        stats_secondary[C_TARGET_RATE_SECONDARY.n] = (
                100 * stats_secondary[C_TARGET_SECONDARY.n] / stats_secondary[C_COUNT_SECONDARY.n].replace(0, EPSILON)
        )

        stats_secondary[C_TARGET_POPULATION_MAIN.n] = (
                100 * stats_secondary[C_TARGET_MAIN.n] /
                (main_target_cnt or EPSILON)
        )
        stats_secondary[C_TARGET_POPULATION_SECONDARY.n] = (
                100 * stats_secondary[C_TARGET_SECONDARY.n] /
                stats_secondary[c_secondary_target_total].replace(0, EPSILON)
        )
        stats_secondary[C_TARGET_STABILITY.n] = (
                (stats_secondary[C_TARGET_POPULATION_SECONDARY.n] - stats_secondary[C_TARGET_POPULATION_MAIN.n]) *
                np.log(
                    stats_secondary[C_TARGET_POPULATION_SECONDARY.n].replace(0, EPSILON) /
                    stats_secondary[C_TARGET_POPULATION_MAIN.n].replace(0, EPSILON)
                ).abs()
        )

    if _sort:
        stats_secondary = stats_secondary.sort_values([split_var_name,] + analyze_vars)
    return stats_secondary


def make_reverse_mapping_pandas(report: DataFrame, var_name_column, var_value_column, mapping: dict) -> DataFrame:
    import pandas as pd

    def f_mapping(row):
        if pd.isnull(row[var_name_column]):
            return None

        var_name = row[var_name_column]
        var_value = row[var_value_column]
        return mapping[var_name][var_value]

    report[var_value_column] = report.apply(f_mapping, axis=1)
    return report


def filter_small_segments_pandas(
        report: DataFrame, filter_dict: dict, unique_cols: List[str]
) -> DataFrame:
    indxs = []
    for col, val in filter_dict.items():
        if col in [C_TARGET_STABILITY.n, C_POPULATION_STABILITY.n]:
            indx = report[col].between(-val, val)
        else:
            indx = report[col] > val
        indxs.append(indx)

    indxs = reduce(lambda x, y: x & y, indxs)
    filtered_segments = report.loc[indxs, unique_cols].drop_duplicates()
    report = filtered_segments.merge(report, on=unique_cols, how='inner').reset_index(drop=True)
    return report
