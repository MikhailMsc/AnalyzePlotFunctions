from analyzer.concentration import calc_concentration_report
from preprocessing import BinningParams



def test_concentration_report_PD():
    import pandas as pd
    df = pd.read_csv('../titanic.csv', sep=',')

    bin_params = BinningParams(
        min_prc=10,
        rnd=0
    )

    concentraion = calc_concentration_report(
        df, target_name='Survived', analyze_vars=['Age', 'Sex'], combo_max=2,
         binning={'Age': bin_params},
        pop_more=10
    )
    print(concentraion)