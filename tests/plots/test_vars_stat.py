from analyzer.plots import plot_var_stat, PlotConfig
from analyzer.preprocessing import BinningParams


def test_var_stat_PD():
    import pandas as pd
    df = pd.read_csv('../titanic.csv', sep=',')

    plot_var_stat(
        df, var_name='Age', target_name='Survived', binning=BinningParams(min_prc=10),
        annotation=True, plot_config=None
    )
