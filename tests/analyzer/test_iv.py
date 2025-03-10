from analyzer.iv import calc_information_value


def test_iv_PD():
    import pandas as pd
    df = pd.read_csv('../titanic.csv', sep=',')
    iv_table = calc_information_value(df, var_name='Age', target_name='Survived', need_binning=True)
    print(iv_table)
    #     17.91385


def test_iv_PL():
    import polars as pl
    df = pl.read_csv('../titanic.csv', separator=',')
    iv_table = calc_information_value(df, var_name='Age', target_name='Survived', need_binning=True)
    print(iv_table)
