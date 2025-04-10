from analyzer.utils.framework_depends import get_unique
from analyzer.utils.framework_depends.columns.is_numeric_column import is_numeric_column
from analyzer.utils.general.types import Series


def validate_binary_target(series: Series) -> None:
    if get_unique(series) != {0, 1}:
        raise Exception('Таргет должен содержать значения 0 и 1, пропуски не допускаются.')


def validate_column_for_binning(series: Series, var_name: str = '') -> None:
    if not is_numeric_column(series):
        error_msg = (
                (f'{var_name}: ' if var_name else '') +
                'Вы хотите применить алгоритм категоризации к нечисловой переменной.'
        )
        raise Exception(error_msg)