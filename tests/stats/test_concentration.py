from analyzer.stats.concentration import calc_concentration_report
from analyzer.preprocessing import BinningParams


def test_concentration_report_PD():
    import pandas as pd
    df = pd.read_csv('../titanic.csv', sep=',')

    bin_params = BinningParams(
        min_prc=10
    )

    concentraion = calc_concentration_report(
        df, target_name='Survived', combo_max=2,
         binning=bin_params
    )
    print(concentraion)


def test_concentration_anomaly():
    import pandas as pd
    df = pd.read_csv('../titanic.csv', sep=',')

    bin_params = BinningParams(
        min_prc=20
    )

    concentraion = calc_concentration_report(
        df, target_name='Survived', combo_max=3,
        analyze_vars=['PassengerId', 'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked'],
        binning=bin_params
    )
    print(concentraion)