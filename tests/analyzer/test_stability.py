from analyzer.stability import calc_stability
from preprocessing.binning import BinningParams


def test_stability_PD():
    import pandas as pd
    df = pd.read_csv('../titanic.csv', sep=',')

    bin_params = BinningParams(
        min_prc=10,
        rnd=0
    )

    stability_stats = calc_stability(
        df, split_var_name='Pclass', analyze_vars=['Age', 'Sex'],
        target_name='Survived', binning={'Age': bin_params}
    )
    print(stability_stats)


def test_stability_PL():
    import polars as pl
    df = pl.read_csv('../titanic.csv', separator=',')

    bin_params = BinningParams(
        min_prc=10,
        rnd=0
    )

    stability_stats = calc_stability(
        df, split_var_name='Pclass', analyze_vars=['Age', 'Sex'],
        target_name='Survived', binning={'Age': bin_params}
    )
    print(stability_stats)
    # -0.63429  ┆ -1.814164