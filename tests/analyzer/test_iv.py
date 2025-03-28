from analyzer.iv import calc_var_information_value, BinningParams, calc_all_information_value


def test_iv_PD():
    import pandas as pd
    df = pd.read_csv('../titanic.csv', sep=',')
    iv_table = calc_var_information_value(df, var_name='Age', target_name='Survived', binning=True)
    print(iv_table)
    #     17.91385


def test_iv_PL():
    import polars as pl
    df = pl.read_csv('../titanic.csv', separator=',')
    iv_table = calc_var_information_value(df, var_name='Age', target_name='Survived', binning=True)
    print(iv_table)


def test_iv_PD_with_params():
    import pandas as pd
    df = pd.read_csv('../titanic.csv', sep=',')

    # bin_params = BinningParams(
    #     cutoffs=[15, 30, 35]
    # )

    bin_params = BinningParams(
        min_prc=10,
        rnd=0
    )

    iv_table = calc_var_information_value(df, var_name='Age', target_name='Survived', binning=bin_params)
    print(iv_table)


def test_iv_PL_full_report():
    import polars as pl
    df = pl.read_csv('../titanic.csv', separator=',')

    total_report_short, total_report = calc_all_information_value(
        df, target_name='Survived', analyze_vars=['Pclass', 'Age', 'Sex'], binning=['Age']
    )

    # print(total_report_short)
    print(total_report)