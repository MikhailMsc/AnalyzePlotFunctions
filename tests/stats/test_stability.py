from analyzer.stats.stability import calc_stability_report
from analyzer.preprocessing.binning import BinningParams


def test_stability_report_PL():
    import polars as pl
    df = pl.read_csv('../titanic.csv', separator=',')

    bin_params = BinningParams(
        min_prc=10,
        rnd=0
    )

    stability_stats = calc_stability_report(
        df, split_var_name='Pclass', analyze_vars=['Age', 'Sex'], combo_max=2,
        target_name='Survived', binning={'Age': bin_params}, split_var_value=1,
        min_part_or_cnt=0.1
    )
    print(stability_stats)
    # -0.63429  ┆ -1.814164



def test_stability_report_PD():
    import pandas as pd
    df = pd.read_csv('../titanic.csv', sep=',')

    bin_params = BinningParams(
        min_prc=10,
        rnd=0
    )

    stability_stats = calc_stability_report(
        df, split_var_name='Pclass', analyze_vars=['Age', 'Sex'], combo_max=2,
        target_name='Survived', binning={'Age': bin_params}, split_var_value=1,
        min_part_or_cnt=0.1
    )
    print(stability_stats)