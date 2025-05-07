from analyzer.plots import plot_cross_vars, PlotConfig
from analyzer.preprocessing import BinningParams


def test_cross_PD():
    import pandas as pd
    df = pd.read_csv('../titanic.csv', sep=',')
    plot_config = PlotConfig(
        colormap='Blues',
        columns_ratios=[6, 1],
        rows_ratios=[1, 6],
        annotation_font_size=8,
        plot_size=(16, 6)
    )

    # _ = plot_cross_vars(
    #     report, var_name_1='Age', var_name_2='Pclass', target_name='Survived', min_population=1.2,
    #     binning={'Age': BinningParams(min_prc=10)}, histogram=False, annotation=True,
    #     plot_config=plot_config, colorbar=False
    # )
    # _ = plot_cross_vars(
    #     report, var_name_1='Age', var_name_2='Pclass', min_population=1.2,
    #     binning={'Age': BinningParams(min_prc=10)}, histogram=True, annotation=True,
    #     plot_config=None, colorbar=True
    # )

    _ = plot_cross_vars(
        df, var_name_1='Age', var_name_2='Pclass', min_population=1.2,
        annotation=True, plot_config=None,  # plot_config
        target_name='Survived',
        binning=BinningParams(min_prc=20),
        histogram=False,
        colorbar=True,
        circles=True,
    )
    print(_)


def test_cross_anomaly():
    import pandas as pd
    df = pd.read_csv('../titanic.csv', sep=',')

    plot_cross_vars(
        df, var_name_1='Age', var_name_2='Parch', min_population=0,
        annotation=True, plot_config=None,
        target_name='Survived',
        binning=BinningParams(min_prc=25),
        histogram=True,
        colorbar=False,
        circles=True
    )
