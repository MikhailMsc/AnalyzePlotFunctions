from analyzer.preprocessing import get_var_cutoffs


def test_get_var_cuttofs_PD_no_target():
    import pandas as pd
    df = pd.read_csv('../titanic.csv', sep=',')
    cutoffs = get_var_cutoffs(df['Age'])
    assert cutoffs == ['-inf', 5, 16, 19, 22, 24, 26, 28, 30, 33, 36, 40, 45, 51, 'inf']


def test_get_var_cuttofs_PL_no_target():
    import polars as pl
    df = pl.read_csv('../titanic.csv', separator=',')
    cutoffs = get_var_cutoffs(df['Age'])
    assert cutoffs == ['-inf', 5, 16, 19, 22, 24, 26, 28, 30, 33, 36, 40, 45, 51, 'inf']


def test_get_var_cuttofs_PD_with_target():
    import pandas as pd
    df = pd.read_csv('../titanic.csv', sep=',')
    cutoffs = get_var_cutoffs(df['Age'], target=df['Survived'])
    assert cutoffs == ['-inf', 6.5, 26.5, 17.5, 21.5, 36.25, 30.75, 28.75, 33.5, 47.5, 40.25, 'inf']

def test_get_var_cuttofs_PL_with_target():
    import polars as pl
    df = pl.read_csv('../titanic.csv', separator=',')
    cutoffs = get_var_cutoffs(df['Age'], target=df['Survived'])
    assert cutoffs == ['-inf', 6.5, 26.5, 17.5, 21.5, 36.25, 30.75, 28.75, 33.5, 47.5, 40.25, 'inf']

