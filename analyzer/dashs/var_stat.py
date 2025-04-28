import re
from typing import List

import ipywidgets as widgets

from IPython.display import display, HTML

from analyzer.plots import plot_var_stat, PlotConfig, plot_cross_vars
from analyzer.plots.empty_plot import plot_empty_area
from analyzer.preprocessing import BinningParams, preprocess_df
from analyzer.stats import calc_iv_report
from analyzer.utils.domain.columns import (
    C_PARENT_MIN, C_PARENT_MAX, C_PARENT_MIN_TR, C_PARENT_MAX_TR,
    C_TOTAL_IV, C_VARNAME
)
from analyzer.utils.domain.validate import get_binary_columns, validate_binary_target
from analyzer.utils.framework_depends import get_columns, get_series_from_df, get_nunique
from analyzer.utils.framework_depends.columns import is_numeric_column
from analyzer.utils.general.types import DataFrame


NONE_VALUE = 'None'


def transaction(method):

    def wrapper(instance, *args, **kwargs):
        if not instance._transaction_start:
            instance._transaction_start = True
            is_main_function = True
        else:
            is_main_function = False

        result = method(instance, *args, **kwargs)

        if is_main_function:
            instance._plot()
            instance._transaction_start = False
        return result

    return wrapper


MAIN_LAYOUT = widgets.Layout(width='27%')
OUTPUT_LAYOUT = widgets.Layout(
    width='95%',
    align_items="center",  # вертикальное выравнивание (если разная высота)
    justify_content="center",  # горизонтальное выравнивание
)


class VarStatDash:

    def __init__(self, df: DataFrame, targets: List[str] = None, bad_cols_start: int = 20):
        self._origin_df = df
        self._all_columns = [NONE_VALUE, ] + sorted(get_columns(df))
        self._cache_iv_reports = dict()  # K = (target, prc_min), V = Dict[Varname, IV]
        self._sort_by_iv = False
        self._transaction_start = False

        if targets:
            self._all_columns = [col for col in self._all_columns if col not in targets]

        not_numeric_columns = [
            col for col in self._all_columns[1:]
            if not is_numeric_column(get_series_from_df(df, col))
        ]
        if not_numeric_columns:
            bad_cols = [col for col, cnt in get_nunique(df, not_numeric_columns).items() if cnt > bad_cols_start]
        else:
            bad_cols = []

        self._all_columns = [col if col not in bad_cols else '❌ ' + col for col in self._all_columns]

        self._x1 = widgets.Dropdown(
            options=self._all_columns,
            description='X1:',
            disabled=False,
            layout=MAIN_LAYOUT
        )
        self._x2 = widgets.Dropdown(
            options=self._all_columns,
            description='X2: ',
            disabled=False,
            layout=MAIN_LAYOUT
        )

        self._min_prc = widgets.IntSlider(
            value=20,
            min=10,
            max=50,
            description='Мин. бакет, %:',
            disabled=False,
            orientation='vertical',
            layout=widgets.Layout(width='15%')
        )

        self._iv_button = widgets.Button(
            description='Сортировать (IV)',
            disabled=False,
            button_style='info',  # 'success', 'info', 'warning', 'danger' or ''
            tooltip='Сортировать (IV)',
            icon='',
            layout=widgets.Layout(width='10%', margin='0 0 0 3%')
        )

        self._plot_output = widgets.Output(layout=OUTPUT_LAYOUT)
        self._table_output = widgets.Output(layout=OUTPUT_LAYOUT)

        if targets:
            for tg in targets:
                validate_binary_target(get_series_from_df(df, tg), tg)
            targets = [NONE_VALUE, ] + targets

        else:
            targets = [NONE_VALUE, ] + get_binary_columns(df)

        self._target = widgets.Dropdown(
            options=targets,
            value=targets[-1 if len(targets) == 2 else 0],
            description='Target: ',
            disabled=False,
            layout=MAIN_LAYOUT

        )

        self._target.observe(lambda _: self._update_target(), names='value')
        self._x1.observe(lambda _: self._update_x1(), names='value')
        self._x2.observe(lambda _: self._update_x2(), names='value')
        self._min_prc.observe(lambda _: self._update_min_prc(), names='value')
        self._iv_button.on_click(lambda _: self._calc_iv_report())

        display(widgets.HBox([self._x1, self._x2, self._target, self._iv_button]))
        display(widgets.HBox([self._min_prc, widgets.VBox([self._plot_output, self._table_output])]))

        self._plot()

    @property
    def _current_x_options(self) -> List[str]:
        if self._sort_by_iv:
            key = (self._target.value, self._min_prc.value)
            return self._cache_iv_reports[key]
        else:
            return self._all_columns

    @transaction
    def _update_target(self):
        self._sort_by_iv = False

        current_target = self._target.value

        x_options = [col for col in self._all_columns if col == NONE_VALUE or col != current_target]
        x1_column = _prepare_column_name(self._x1.value)

        self._x1.options = x_options
        if x1_column == current_target:
            self._x1.value = NONE_VALUE
        else:
            self._x1.value = [col for col in self._x1.options if _prepare_column_name(col) == x1_column][0]

        x2_column = _prepare_column_name(self._x2.value)
        self._x2.options = x_options
        if x2_column == current_target:
            self._x2.value = NONE_VALUE
        else:
            self._x2.value = [col for col in self._x2.options if _prepare_column_name(col) == x2_column][0]

    @transaction
    def _update_x1(self):
        if self._x1.value == NONE_VALUE:
            pass
        elif self._x1.value == self._x2.value:
            self._x2.value = NONE_VALUE

    @transaction
    def _update_x2(self):
        if self._x2.value == NONE_VALUE:
            pass
        elif self._x2.value == self._x1.value:
            self._x1.value = NONE_VALUE

    @transaction
    def _update_min_prc(self):
        was_sorted = self._sort_by_iv
        self._sort_by_iv = False
        self._iv_button.disabled = False

        if was_sorted:
            current_target = self._target.value
            x_options = [col for col in self._all_columns if col == NONE_VALUE or col != current_target]
            x1_column = _prepare_column_name(self._x1.value)
            self._x1.options = x_options
            self._x1.value = [col for col in self._x1.options if _prepare_column_name(col) == x1_column][0]

            x2_column = _prepare_column_name(self._x2.value)
            self._x2.options = x_options
            self._x2.value = [col for col in self._x2.options if _prepare_column_name(col) == x2_column][0]

    def _plot(self):
        with self._plot_output:
            self._plot_output.clear_output()
            report = _plot_func(
                self._origin_df,
                _prepare_column_name(self._x1.value),
                _prepare_column_name(self._x2.value),
                self._target.value, self._min_prc.value,
                True
            )

        with self._table_output:
            self._table_output.clear_output()
            if report is not None:
                if self._x2.value != NONE_VALUE and self._x1.value != NONE_VALUE:
                    report.drop(
                        columns=[C_PARENT_MIN.n, C_PARENT_MAX.n, C_PARENT_MIN_TR.n, C_PARENT_MAX_TR.n],
                        inplace=True
                    )
                elif self._x2.value != NONE_VALUE or self._x1.value != NONE_VALUE:
                    report.drop(
                        columns=[C_VARNAME.n],
                        inplace=True
                    )
                display(HTML(report.to_html()))

    @transaction
    def _calc_iv_report(self):
        assert self._target.value != NONE_VALUE
        key_report = (self._target.value, self._min_prc)
        self._sort_by_iv = True

        if key_report not in self._cache_iv_reports:
            analyze_vars = [
                clear_col for col in self._x1.options
                if
                    col != NONE_VALUE and
                    '❌' not in col and
                    (clear_col := _prepare_column_name(col)) != self._target.value
            ]
            stop_cols = [col for col in self._x1.options if '❌' in col]

            total_iv_stats, _ = calc_iv_report(
                self._origin_df, self._target.value, analyze_vars, None,
                BinningParams(min_prc=self._min_prc.value)
            )
            vars_names = (
                    total_iv_stats[C_TOTAL_IV.n].map(lambda x: f'({str(round(x, 1))}) ') +
                    total_iv_stats[C_VARNAME.n]
            ).to_list() + stop_cols

            self._cache_iv_reports[key_report] = [NONE_VALUE, ] + vars_names

        self._x1.value = NONE_VALUE
        self._x1.options = self._cache_iv_reports[key_report]
        self._x2.value = NONE_VALUE
        self._x2.options = self._cache_iv_reports[key_report]
        self._iv_button.disabled = True


def _plot_func(df, var1, var2, target, min_prc, circles):
    report = None
    if var1 == var2 == NONE_VALUE and target != NONE_VALUE:
        # Гистограмма таргета
        plot_config = PlotConfig(plot_size=(10, 4.5))

        report = plot_var_stat(
            df, var_name=target,
            binning=False,
            plot_config=plot_config  # PlotConfig
        )
    elif [var1, var2].count(NONE_VALUE) == 1:
        # Гистограмма переменной
        plot_config = PlotConfig(plot_size=(10, 4.5))
        var = var1 if var1 != NONE_VALUE else var2
        report = plot_var_stat(
            df, var, None if target == NONE_VALUE else target,
            binning=BinningParams(min_prc=min_prc),
            plot_config=plot_config
        )

    elif var1 != NONE_VALUE and var2 != NONE_VALUE:
        # Cross Plot
        df = preprocess_df(
            df, [var1, var2], None, target, BinningParams(min_prc=min_prc),
            None, False, False
        )

        cnt_groups = get_nunique(df, [var1, var2])
        single_cols = [col for col, cnt in cnt_groups.items() if cnt == 1]
        if len(single_cols) == 1:
            plot_config = PlotConfig(plot_size=(5, 5), annotation_font_size=10)
        else:
            if max(cnt_groups.values()) > 4:
                plot_config = PlotConfig(plot_size=(6, 6), annotation_font_size=8.6)
            else:
                plot_config = PlotConfig(plot_size=(7.3, 7.3), annotation_font_size=8)

        report = plot_cross_vars(
            df, var1, var2, None if target == NONE_VALUE else target,
            binning=False,
            colorbar=False,
            plot_config=plot_config,
            circles=circles
        )
    else:
        plot_config = PlotConfig(plot_size=(6, 6), annotation_font_size=20)
        plot_empty_area('Введите необходимые данные.', plot_config)

    return report


_reg_iv = re.compile('\\([0-9]+(.[0-9]+){0,1}\\) ')


def _prepare_column_name(input_name: str) -> str:
    if input_name == NONE_VALUE:
        return input_name

    if match := _reg_iv.match(input_name):
        input_name = input_name[match.regs[0][-1]:]

    return input_name.strip('❌ ')
