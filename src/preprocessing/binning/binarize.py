from dataclasses import dataclass
from typing import List, Union

from utils.domain.const import MIN, MAX, MISSING, NOT_MISSING
from utils.domain.validate import validate_column_for_binning
from utils.general.types import Series, FrameWork, get_framework_from_series

from utils.general.utils import pretty_number

from .cutoffs import get_var_cutoffs


@dataclass
class BinningParams:
    # Предопределенный бинниг
    cutoffs: List[Union[int, float]] = None

    # Расчетный биннинг
    min_prc: float = 5.0  # Минимальный размер группы
    rnd: int = None       # Округление перед биннингом


def binarize_series(
        variable: Series, cutoffs: list = None, target: Series = None, min_prc: float = 5.0,
        rnd: int = None, validate_target: bool = True, _var_name: str = ''
) -> Series:
    if cutoffs is None:
        cutoffs = get_var_cutoffs(variable, target, min_prc, rnd, validate_target)
    else:
        validate_column_for_binning(variable, _var_name)
        cutoffs = [MIN, ] + cutoffs + [MAX,]

    framework = get_framework_from_series(variable)
    func = _MAP_FRAMEWORK_FUNC[framework]
    return func(variable, cutoffs)


def _apply_cutoffs_pandas(series: Series, cutoffs: list):
    import pandas as pd
    from pandas.api.types import CategoricalDtype as PandasCategoricalDtype
    import numpy as np

    has_missing = series.isnull().sum() > 0
    if cutoffs == [MIN, MAX]:
        if has_missing:
            series = series.fillna(MISSING)
            categories = [MISSING, NOT_MISSING]
        else:
            categories = [NOT_MISSING]
        series[series != MISSING] = NOT_MISSING
        category_type = PandasCategoricalDtype(categories=categories, ordered=True)
        series = series.astype(category_type)

    else:
        labels = []
        for i, p in enumerate(cutoffs):
            p = pretty_number(p)
            if i == 0:
                continue
            elif i == 1:
                labels.append(f'<= {p}')
            else:
                labels.append(f'({pretty_number(cutoffs[i-1])}; {p}]')

        labels[-1] = f'> {pretty_number(cutoffs[-2])}'

        cutoffs[0] = -np.inf
        cutoffs[-1] = np.inf

        series = pd.cut(series, bins=cutoffs, labels=labels, right=True, ordered=True)
        if has_missing:
            series = series.cat.add_categories(MISSING).fillna(MISSING)
            series = series.cat.reorder_categories([MISSING,] + labels)
    return series


def _apply_cutoffs_polars(series: Series, cutoffs: list):
    import polars as pl

    has_missing = series.is_null().sum() > 0
    if cutoffs == [MIN, MAX]:
        if has_missing:
            series = series.fill_null(MISSING)
            categories = [MISSING, NOT_MISSING]
        else:
            categories = [NOT_MISSING]
        series[series != MISSING] = NOT_MISSING
        category_type = pl.Enum(categories)
        series = series.cast(category_type)

    else:
        labels = []
        for i, p in enumerate(cutoffs):
            p = pretty_number(p)
            if i == 0:
                continue
            elif i == 1:
                labels.append(f'<= {p}')
            else:
                labels.append(f'({pretty_number(cutoffs[i - 1])}; {p}]')

        labels[-1] = f'> {pretty_number(cutoffs[-2])}'
        cutoffs = cutoffs[1:-1]

        # labels = labels,
        series = series.cut(breaks=cutoffs, labels=labels, left_closed=False)
        if has_missing:
            categories = pl.Enum([MISSING, ] + labels)
            series = series.cast(categories)
            series = series.fill_null(MISSING)

    return series


def _apply_cutoffs_spark(series: Series, cutoffs: list):
    raise NotImplementedError


_MAP_FRAMEWORK_FUNC = {
    FrameWork.pandas: _apply_cutoffs_pandas,
    FrameWork.polars: _apply_cutoffs_polars,
    FrameWork.spark: _apply_cutoffs_spark,
}



