from typing import List

from analyzer.utils.framework_depends import get_unique, get_columns, get_series_from_df
from analyzer.utils.framework_depends.columns.is_numeric_column import is_numeric_column
from analyzer.utils.general.types import Series, DataFrame


def validate_binary_target(series: Series, var_name: str = '') -> None:
    if get_unique(series) != {0, 1}:
        error_msg = (
                (f'{var_name}: ' if var_name else '') +
                'Таргет должен содержать исключительно значения 0 и 1, пропуски не допускаются.'
        )
        raise Exception(error_msg)


def validate_column_for_binning(series: Series, var_name: str = '', raise_exception: bool = True) -> bool:
    if not is_numeric_column(series):
        if raise_exception:
            error_msg = (
                    (f'Variable = {var_name}: ' if var_name else '') +
                    'Вы хотите применить алгоритм категоризации к нечисловой переменной.'
            )
            raise Exception(error_msg)
        else:
            return False
    return True


def get_binary_columns(df: DataFrame) -> List[str]:
    columns = get_columns(df)
    binary_columns = []
    for col in columns:
        ser = get_series_from_df(df, col)
        if get_unique(ser) == {0, 1}:
            binary_columns.append(col)
    return binary_columns
