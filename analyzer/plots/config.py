from dataclasses import dataclass, fields
from typing import Literal, Tuple, List, Union, Self

from matplotlib import pyplot as plt

AVAILABLE_STYLES = plt.style.available[:]


@dataclass
class PlotConfig:
    """
    Args:
        style: str                          Стиль графика. Список доступных значений можно узнать через:
                                            from matplotlib import pyplot as plt
                                            print(plt.style.available)
        plot_size: Tuple[float, float]      Размер изображения (X, Y)

        colormap: str                       Название цветовой палитры
                                            https://www.practicalpythonfordatascience.com/ap_seaborn_palette
        color: str                          Цвет 1
        color2: str                         Цвет 2

        annotation_delta: float             Смещение аннотации по вертикали
        annotation_font_size: float         Размер шрифта аннотации

        title: str                          Название графика
        title_size: float                   Размер шрифта для названия графика
        title_pad: float                    Отступт между графиком и названием

        xlabel: str                         Название оси X
        xlabel_size: float                  Размер названия оси Х
        xtick_size: float                   Размер делений на оси Х
        x_rotation: float                   Поворот подписей на оси Х
        x_grid: bool                        Отображение сетки X на графике
        xmax: float                         Максимальное значение Х
        xmin: float                         Минимальное значение Х

        ylabel: str                         Название оси Y
        ylabel_size: float                  Размер названия оси Y
        ytick_size: float                   Размер делений на оси Y
        y_rotation: float                   Поворот подписей на оси Y
        y_grid: bool                        Отображение сетки Y на графике
        ymax: float                         Максимальное значение Y
        ymin: float                         Минимальное значение Y

        y2label: str                        Название оси Y2
        y2_grid: bool                       Отображение сетки Y2 на графике
        y2max: float                        Максимальное значение Y2
        y2min: float                        Минимальное значение Y2

        grid_widths: float                  Толщина линий сетки графика
        bar_width: float                    Толщина столбцов на столбчатых диаграммах
        side_grid: bool                     Отображение сетки на вспомогательных графиках

        cbar_location: 'bottom', 'top', 'left', 'right
                                            Место локации колорбар. Актуально не для всех графиков.
        cbar_width: float                   Толщина колорбара
        cbar_pad: float                     Размер отступа колорбара от графика

        -- Нужно только для сложных композиций графиков.
        columns_ratios: List[float]         Отношение между колонками сетки графика
        rows_ratios: List[float]            Отношение между строками сетки графика
        empty_space: List[float]            Размер промежутков между ячейками сетки графика
    """
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

    ylabel: str = None
    ylabel_size: float = 15
    ytick_size: float = 10
    y_rotation: float = None
    y_grid: bool = True
    ymax: float = None
    ymin: float = None

    y2label: str = None
    y2_grid: bool = False
    y2max: float = None
    y2min: float = None

    columns_ratios: List[float] = None
    rows_ratios: List[float] = None
    empty_space: float = None
    grid_widths: float = None
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
