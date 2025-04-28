from dataclasses import dataclass, fields
from typing import Literal, Tuple, List, Union, Self

from matplotlib import pyplot as plt

AVAILABLE_STYLES = plt.style.available[:]


@dataclass
class PlotConfig:
    style: Literal[*AVAILABLE_STYLES] = None
    plot_size: Tuple[float, float] = None

    colormap: str = None
    color: str = None
    color2: str = None

    annotation_delta: float = None
    annotation_font_size: float = None

    title: str = None
    title_size: float = 18
    title_pad: float = 0.0

    xlabel: str = ''
    xlabel_size: float = 15
    xtick_size: float = 10
    x_rotation: float = None
    x_grid: bool = True
    xmax: float = None
    xmin: float = None

    ylabel: str = ''
    ylabel_size: float = 15
    ytick_size: float = 10
    y_rotation: float = None
    y_grid: bool = True
    ymax: float = None
    ymin: float = None

    y2label: str = ''
    y2_grid: bool = False
    y2max: float = None
    y2min: float = None

    x_to_top: bool = False
    x_inverse: bool = False
    y_to_right: bool = False
    y_inverse: bool = False

    columns_ratios: List[float] = None
    rows_ratios: List[float] = None
    empty_space: float = None
    grig_widths: float = None
    bar_width: float = 0.99
    side_grid: bool = None

    cbar_location: str = None
    cbar_width: float = 5
    cbar_pad: float = 1

    def merge(self, other) -> Self:
        self_fields = fields(self)
        for f in self_fields:
            if (val := getattr(other, f.name)) is not None:
                setattr(self, f.name, val)
        return self
