from typing import List, Tuple, Dict, TypeVar, Literal

from analyzer.utils.general.types import DataFrame

from ..unique import get_unique
from ..columns import get_columns, get_series_from_df, drop_columns
from ..mapping import map_elements_df

VarName = TypeVar('VarName')
OrderValue = TypeVar('OrderValue')
OldValue = TypeVar('OldValue')
MinValue = TypeVar('MinValue')
MaxValue = TypeVar('MaxValue')

ReverseMapDictMultiVars = Dict[VarName, Dict[OrderValue, OldValue]]
MinMaxVarsValues = Dict[VarName, Tuple[MinValue, MaxValue]]


def encode_df(
        df: DataFrame, encode_vars: List[str] = None, ignore_vars: List[str] = None,
        bad_vars_behavior: Literal['drop', 'raise', 'skip'] = False,
        _start_bad_var: int = 200
) -> Tuple[DataFrame, ReverseMapDictMultiVars, MinMaxVarsValues]:
    if encode_vars is None:
        encode_vars = get_columns(df)

    if ignore_vars:
        encode_vars = [col for col in encode_vars if col not in ignore_vars]

    map_vars = dict()
    reverse_map_vars = dict()
    min_max_values = dict()
    dropped_columns = []

    for col in encode_vars:
        col_values = get_unique(get_series_from_df(df, col))
        col_min_max_id = (0, len(col_values) - 1)

        if col_min_max_id[1] > _start_bad_var - 1:
            if bad_vars_behavior == 'drop':
                dropped_columns.append(col)
                continue
            elif bad_vars_behavior == 'raise':
                msg = (f'encode_df: В колонке "{col}" много значений ({min_max_values[col][1] + 1} > 200). '
                       f'Вероятно не стоит применять к ней encoding. '
                       f'Но вы можете сделать это принудительно, использовав параметр _force=True')
                raise Exception(msg)
            elif bad_vars_behavior == 'skip':
                continue
            else:
                raise Exception(
                    f'encode_df: Неизвестное значение bad_vars_behavior = {bad_vars_behavior}, '
                    f'доступные значения = drop, raise, skip.'
                )

        min_max_values[col] = col_min_max_id
        map_values = {val: i for i, val in enumerate(col_values)}
        reverse_map_values = {i: val for val, i in map_values.items()}

        map_vars[col] = map_values
        reverse_map_vars[col] = reverse_map_values

    if dropped_columns:
        df = drop_columns(df, dropped_columns)

    df = map_elements_df(df, map_vars, 'int16')
    return df, reverse_map_vars, min_max_values




