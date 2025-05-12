from analyzer.preprocessing import binarize_series


def test_binning_PD():
    import pandas as pd
    df = pd.read_csv('../titanic.csv', sep=',')
    print('CNT MISSING: ', df['Age'].isnull().sum())
    bin_ser = binarize_series(df['Age'])
    print(bin_ser)


def test_binning_PL():
    import polars as pl
    df = pl.read_csv('../titanic.csv', separator=',')
    bin_ser = binarize_series(df['Age'])
    print(bin_ser)
