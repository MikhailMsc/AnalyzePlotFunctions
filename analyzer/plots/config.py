from dataclasses import dataclass
from typing import Literal, Tuple

from matplotlib import pyplot as plt

AVAILABLE_STYLES = plt.style.available[:]


@dataclass
class PlotConfig:
    style: Literal[*AVAILABLE_STYLES] = 'ggplot'
    figure_size: Tuple[int, int] = (20, 20)
