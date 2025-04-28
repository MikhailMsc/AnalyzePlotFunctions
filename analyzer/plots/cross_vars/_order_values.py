import re
from typing import List, Union

from analyzer.utils.domain.columns import C_TARGET_RATE, C_POPULATION
from analyzer.utils.domain.const import MISSING, NOT_MISSING


def get_order_vars_values(df, has_target) -> List[str]:
    values = df['Var1_Value'].to_list()
    ordered_values = []

    if MISSING in values:
        ordered_values.append(MISSING)
        values = [v for v in values if v != MISSING]

    if NOT_MISSING in values:
        ordered_values.append(NOT_MISSING)
        values = [v for v in values if v != NOT_MISSING]
        assert not values
        return ordered_values

    types_values = set([type(v) for v in values])
    assert len(types_values) == 1
    types_values = types_values.pop()

    if types_values in [float, int]:
        ordered_values.extend(sorted(values))
        return ordered_values

    if digits := _order_digits(values):
        ordered_values.extend(digits)
        return ordered_values

    if buckets := _order_buckets(values):
        ordered_values.extend(buckets)
        return ordered_values

    import numpy as np
    indx_not_missing = df['Var1_Value'] != MISSING
    order_by = df.loc[indx_not_missing, C_TARGET_RATE.n if has_target else C_POPULATION.n].to_list()
    order_indx = np.argsort(order_by)
    values = [values[i] for i in order_indx]
    ordered_values.extend(values)
    return ordered_values


_re_digit = re.compile('-{0,1}[0-9_]+(\\.[0-9_]+){0,1}')
_reg_min_bucket = re.compile('<= (-{0,1}[0-9_]+(\\.[0-9_]+){0,1}){1}')
_reg_middle_bucket = re.compile('\\((-{0,1}[0-9_]+(\\.[0-9_]+){0,1}){1}; (-{0,1}[0-9_]+(\\.[0-9_]+){0,1}){1}]')
_reg_max_bucket = re.compile('> (-{0,1}[0-9_]+(\\.[0-9_]+){0,1}){1}')


def _order_digits(values: List[str]) -> Union[List[str], None]:
    reg_values = []
    for val in values:
        reg = _re_digit.match(val)
        if not reg:
            return None
        reg_values.append(float(reg.group(0)))

    import numpy as np
    order_indx = np.argsort(reg_values)
    values = [values[i] for i in order_indx]
    return values


def _order_buckets(values: List[str]) -> Union[List[str], None]:
    reg_values = []
    for val in values:
        reg = _reg_min_bucket.match(val) or _reg_middle_bucket.match(val) or _reg_max_bucket.match(val)
        if not reg:
            return None
        else:
            if '(' in val:
                val = float(reg.group(3))
            elif '>' in val:
                val = float(reg.group(1)) + 1
            else:
                val = float(reg.group(1))
            reg_values.append(val)

    import numpy as np
    order_indx = np.argsort(reg_values)
    values = [values[i] for i in order_indx]
    return values
