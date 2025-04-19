from typing import List, Tuple

import seaborn as sns

from analyzer.utils.general.types import DataFrame
from analyzer.utils.framework_depends import get_series_from_df, get_shape, series_to_list
from analyzer.utils.domain.columns import C_POPULATION, C_TARGET_RATE

from ..config import PlotConfig


def prepare_plot_config(
        report: DataFrame, xlabel: str, has_target: bool,
        plot_config: PlotConfig = None
) -> Tuple[PlotConfig, List]:
    if plot_config is None:
        plot_config = PlotConfig()
        plot_config.plot_size = (10, 15)

    plot_config.xlabel = xlabel
    plot_config.ylabel = 'Population(%)'
    plot_config.title_pad = -0.03

    if has_target:
        plot_config.y2label = 'Target Rate(%)'
        if plot_config.color2 is None:
            plot_config.color2 = 'red'

    if plot_config.colormap is None:
        plot_config.colormap = 'Blues'

    if plot_config.color is None:
        palette = sns.color_palette(plot_config.colormap, get_shape(report)[0]).as_hex()
        order_by = get_series_from_df(report, C_TARGET_RATE.n if has_target else C_POPULATION.n)
        order_by = series_to_list(order_by)
        palette = _reorder_palette(palette, order_by)
    else:
        palette = []

    plot_config.annotation_delta = 0
    return plot_config, palette


def _reorder_palette(palette: List, order_by: List) -> List:
    assert len(palette) == len(order_by)

    import numpy as np
    order_colors = sorted(
        [(true_pos, value_pos) for value_pos, true_pos in enumerate(np.argsort(order_by))],
        key=lambda x: x[0]
    )
    palette = [palette[pos] for _, pos in order_colors]
    return palette
