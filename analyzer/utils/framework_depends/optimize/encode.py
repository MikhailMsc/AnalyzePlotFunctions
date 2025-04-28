from typing import List, Tuple, Dict, TypeVar

from analyzer.utils.general.types import DataFrame

from ..unique import get_unique
from ..columns import get_columns, get_series_from_df
from ..mapping import map_elements_df


VarName = TypeVar('VarName')
OrderValue = TypeVar('OrderValue')
OldValue = TypeVar('OldValue')
MinValue = TypeVar('MinValue')
MaxValue = TypeVar('MaxValue')

ReverseMapDictMultiVars = Dict[VarName, Dict[OrderValue, OldValue]]
MinMaxVarsValues = Dict[VarName, Tuple[MinValue, MaxValue]]


def encode_df(
        df: DataFrame, encode_vars: List[str] = None, ignore_vars: List[str] = None, _force: bool = False
) -> Tuple[DataFrame, ReverseMapDictMultiVars, MinMaxVarsValues]:
    if encode_vars is None:
        encode_vars = get_columns(df)

    if ignore_vars:
        encode_vars = [col for col in encode_vars if col not in ignore_vars]

    map_vars = dict()
    reverse_map_vars = dict()
    min_max_values = dict()

    for col in encode_vars:
        col_values = get_unique(get_series_from_df(df, col))
        min_max_values[col] = (0, len(col_values) - 1)

        if not _force and min_max_values[col][1] > 200 - 1:
            msg = (f'encode_df: В колонке {col} много значений ({min_max_values[col][1] + 1} > 200). '
                   f'Вероятно не стоит применять к ней encoding.'
                   f'Но вы можете сделать это принудительно, использовав параметр _force=True')
            raise Exception(msg)

        map_values = {val: i for i, val in enumerate(col_values)}
        reverse_map_values = {i: val for val, i in map_values.items()}

        map_vars[col] = map_values
        reverse_map_vars[col] = reverse_map_values

    df = map_elements_df(df, map_vars, 'int16')
    return df, reverse_map_vars, min_max_values




