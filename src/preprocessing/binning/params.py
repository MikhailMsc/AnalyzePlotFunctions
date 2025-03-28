from dataclasses import dataclass
from typing import List, Union


@dataclass
class BinningParams:
    # Расчетный биннинг
    min_prc: float = 5.0  # Минимальный размер группы
    rnd: int = None       # Округление катофоффов

    # Предопределенный бинниг
    cutoffs: List[Union[int, float]] = None


default_bin_params = BinningParams()
