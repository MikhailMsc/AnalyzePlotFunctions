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
    #     df, var_name_1='Age', var_name_2='Pclass', target_name='Survived', min_population=1.2,
    #     binning={'Age': BinningParams(min_prc=10)}, histogram=False, annotation=True,
    #     plot_config=plot_config, colorbar=False
    # )
    _ = plot_cross_vars(
        df, var_name_1='Age', var_name_2='Pclass', min_population=1.2,
        binning={'Age': BinningParams(min_prc=10)}, histogram=True, annotation=True,
        plot_config=None, colorbar=True
    )
    print(_)



"""
        data: DataFrame, var_name_1: str, var_name_2: str, target_name: str = None,
        map_values: MapDictMultiVars = None, binning: BinningParamsMultiVars = True,
        annotation: bool = True, plot_config: PlotConfig = None
"""