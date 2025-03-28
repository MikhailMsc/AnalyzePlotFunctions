from typing import Union, List

import numpy as np

from utils.domain.columns import (
    C_COUNT_SECONDARY, C_TARGET_SECONDARY, C_TARGET_MAIN, C_COUNT_MAIN, C_POPULATION_MAIN,
    C_POPULATION_SECONDARY, C_POPULATION_STABILITY, C_TARGET_RATE_MAIN, C_TARGET_RATE_SECONDARY,
    C_TARGET_POPULATION_MAIN, C_TARGET_POPULATION_SECONDARY, C_TARGET_STABILITY
)
from utils.domain.const import EPSILON
from utils.framework_depends import fill_missing_df
from utils.general.types import DataFrame

from ._const import c_secondary_count_total, c_secondary_target_total


def calc_stability(
        df: DataFrame, split_var_name: str, analyze_vars: List[str],
        target_name: Union[str, None], split_var_value, _sort: bool = True
) -> DataFrame:
    import polars as pl
    if split_var_value is not None:
        main_df = df.filter(pl.col(split_var_name) == split_var_value)
    else:
        main_df = df

    grp_cols = [split_var_name, ] + analyze_vars
    if target_name:
        stats_secondary = df.group_by(grp_cols).agg(
            pl.col(target_name).count().alias(C_COUNT_SECONDARY.n),
            pl.col(target_name).sum().alias(C_TARGET_SECONDARY.n)
        )

        stats_main = main_df.group_by(analyze_vars).agg(
            pl.col(target_name).count().alias(C_COUNT_MAIN.n),
            pl.col(target_name).sum().alias(C_TARGET_MAIN.n)
        )

        general_stats_secondary = df.group_by([split_var_name]).agg(
            pl.col(target_name).count().alias(c_secondary_count_total),
            pl.col(target_name).sum().alias(c_secondary_target_total)
        )

    else:
        stats_secondary = df.group_by(grp_cols).agg(pl.count().alias(C_COUNT_SECONDARY.n))
        stats_main = main_df.group_by(analyze_vars).agg(pl.count().alias(C_COUNT_MAIN.n))
        general_stats_secondary = df.group_by([split_var_name]).agg(pl.count().alias(c_secondary_count_total))

    main_total_cnt = stats_main[C_COUNT_MAIN.n].sum()
    main_target_cnt = stats_main[C_TARGET_MAIN.n].sum()

    all_groups = df.select(analyze_vars).unique()
    all_splits = df.select(split_var_name).unique()

    all_groups_splits = all_splits.join(all_groups, how='cross')
    del all_splits

    stats_main = all_groups.join(stats_main, on=analyze_vars, how='left')
    stats_secondary = all_groups_splits.join(stats_secondary, on=grp_cols, how='left')
    del all_groups, all_groups_splits

    stats_secondary = stats_secondary.join(stats_main, on=analyze_vars, how='inner')
    del stats_main

    stats_secondary = stats_secondary.join(general_stats_secondary, how='inner', on=[split_var_name])
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
    #     stats_secondary = stats_secondary.with_columns(
    #         pl.when(pl.col(split_var_name) == split_var_value).
    #         then(pl.lit(None)).otherwise(pl.col(C_COUNT_SECONDARY.n)).alias(C_COUNT_SECONDARY.n),
    #         pl.when(pl.col(split_var_name) == split_var_value).
    #         then(pl.lit(None)).otherwise(pl.col(C_TARGET_SECONDARY.n)).alias(C_TARGET_SECONDARY.n),
    #     )

    # Статистики общей популяции
    stats_secondary = stats_secondary.with_columns(
        (100 * pl.col(C_COUNT_MAIN.n) / main_total_cnt).alias(C_POPULATION_MAIN.n),
        (100 * pl.col(C_COUNT_SECONDARY.n) / c_secondary_count_total).alias(C_POPULATION_SECONDARY.n),
    )

    stats_secondary = stats_secondary.with_columns(
        (
            (pl.col(C_POPULATION_SECONDARY.n) - pl.col(C_POPULATION_MAIN.n)) * (
                pl.when(pl.col(C_POPULATION_SECONDARY.n) == 0).then(EPSILON).otherwise(pl.col(C_POPULATION_SECONDARY.n)) /
                pl.when(pl.col(C_POPULATION_MAIN.n) == 0).then(EPSILON).otherwise(pl.col(C_POPULATION_MAIN.n))
            ).log().abs()
        ).alias(C_POPULATION_STABILITY.n)
    )

    # Статистики популяции таргета
    if target_name:
        stats_secondary = stats_secondary.with_columns(
            (
                100 * pl.col(C_TARGET_MAIN.n) /
                pl.when(pl.col(C_COUNT_MAIN.n) == 0).then(EPSILON).otherwise(pl.col(C_COUNT_MAIN.n))
            ).alias(C_TARGET_RATE_MAIN.n),
            (
                100 * pl.col(C_TARGET_SECONDARY.n) /
                pl.when(pl.col(C_COUNT_SECONDARY.n) == 0).then(EPSILON).otherwise(pl.col(C_COUNT_SECONDARY.n))
            ).alias(C_TARGET_RATE_SECONDARY.n),
            (
                100 * pl.col(C_TARGET_MAIN.n) / (main_target_cnt or EPSILON)
            ).alias(C_TARGET_POPULATION_MAIN.n),
            (
                100 * pl.col(C_TARGET_SECONDARY.n) /
                pl.when(pl.col(c_secondary_target_total) == 0).then(EPSILON).otherwise(pl.col(c_secondary_target_total))
            ).alias(C_TARGET_POPULATION_SECONDARY.n),

        )

        stats_secondary = stats_secondary.with_columns(
            (
                (pl.col(C_TARGET_POPULATION_SECONDARY.n) - pl.col(C_TARGET_POPULATION_MAIN.n)) * (
                    pl.when(pl.col(C_TARGET_POPULATION_SECONDARY.n) == 0).
                        then(EPSILON).otherwise(pl.col(C_TARGET_POPULATION_SECONDARY.n)) /
                    pl.when(pl.col(C_TARGET_POPULATION_MAIN.n) == 0).then(EPSILON).otherwise(pl.col(C_TARGET_POPULATION_MAIN.n))
                ).log().abs()
            ).alias(C_TARGET_STABILITY.n)
        )

    if _sort:
        stats_secondary = stats_secondary.sort(split_var_name, *analyze_vars)
    return stats_secondary

