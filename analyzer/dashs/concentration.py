from dataclasses import dataclass
from functools import reduce
from typing import List

import ipywidgets as widgets
from IPython.display import display, HTML
from matplotlib_venn import venn3
from matplotlib import pyplot as plt
import seaborn as sns
import warnings

from analyzer.plots import PlotConfig, plot_var_stat
from analyzer.preprocessing import BinningParams, preprocess_df
from analyzer.stats import calc_concentration_report
from analyzer.utils.domain.columns import C_GROUP_IV
from analyzer.utils.framework_depends import get_columns, get_series_from_df, get_nunique, get_sub_df
from analyzer.utils.framework_depends.columns import is_numeric_column
from analyzer.utils.general.types import DataFrame, get_framework_from_dataframe, FrameWork

warnings.filterwarnings('ignore', module='matplotlib_venn')

NONE_VALUE = 'None'
TOP_ITEMS_LAYOUT = widgets.Layout(width='33%')
MAX_COMBO = 5


class ConcentrationDash:

    def __init__(
            self, df: DataFrame, targets_names: List[str], combo_min: int = 1, combo_max: int = 2,
            min_bin: float = 20.0, bad_cols_start: int = 30, max_cached_reports = 3
    ):
        self._origin_df = df
        self._cache_report = dict()  # K = (min_bin, combo_min, combo_max, target_name), V = Report
        self._max_cached_reports = max_cached_reports

        var_columns = [col for col in get_columns(df) if col not in targets_names]
        not_numeric_columns = [
            col for col in var_columns
            if not is_numeric_column(get_series_from_df(df, col))
        ]
        if not_numeric_columns:
            bad_columns = [col for col, cnt in get_nunique(df, not_numeric_columns).items() if cnt > bad_cols_start]
        else:
            bad_columns = []
        self._var_columns = [col for col in var_columns if col not in bad_columns]

        assert combo_min <= MAX_COMBO
        assert combo_max <= MAX_COMBO
        assert combo_max >= combo_min

        self._cnt_combo = widgets.IntRangeSlider(
            value=[combo_min, combo_max],
            min=1,
            max=5,
            description='Комбо:',
            disabled=False,
            style={'description_width': 'initial'},
            layout=widgets.Layout(width='20%'),
        )

        assert (min_bin >= 10) and (min_bin <= 40)
        self._min_bucket = widgets.IntSlider(
            value=min_bin,
            min=10,
            max=40,
            description='Мин. бакет, %:',
            disabled=False,
            style={'description_width': 'initial'},
            layout=widgets.Layout(width='25%'),
        )
        self._min_iv = widgets.FloatSlider(
            value=0,
            min=0,
            max=100,
            description='Мин. IV:',
            disabled=False,
            style={'description_width': 'initial'},
            layout=widgets.Layout(width='25%'),
        )

        targets_names = sorted(targets_names)
        self._target = widgets.Dropdown(
            value=targets_names[0],
            options=targets_names,
            description='Таргет:',
            disabled=False,
            style={'description_width': 'initial'},
            layout=widgets.Layout(width='30%'),
        )

        self._target = widgets.Dropdown(
            value=targets_names[0],
            options=targets_names,
            description='Таргет:',
            disabled=False,
            style={'description_width': 'initial'},
            layout=TOP_ITEMS_LAYOUT,
        )

        self._segment = widgets.Dropdown(
            value=NONE_VALUE,
            options=[NONE_VALUE],
            description='Сегмент:',
            disabled=False,
            style={'description_width': 'initial'},
            layout=widgets.Layout(width='87%'),
        )
        self._calc_button = widgets.Button(
            description='Рассчитать сегменты',
            disabled=False,
            button_style='info',  # 'success', 'info', 'warning', 'danger' or ''
            tooltip='Рассчитать сегменты',
            style={'description_width': 'initial'},
            icon='',
            layout=widgets.Layout(width='13%', margin='0 0 0 1%')
        )
        self._plot_output = widgets.Output(layout=widgets.Layout(
            width='80%',
            align_items="center",  # вертикальное выравнивание (если разная высота)
            justify_content="center",  # горизонтальное выравнивание,
            margin='0 0 0 5%'
        ))
        self._table_output = widgets.Output(layout=widgets.Layout(
            width='100%',
            align_items="center",  # вертикальное выравнивание (если разная высота)
            justify_content="center",  # горизонтальное выравнивание,
        ))
        self._with_dop_plots = widgets.Checkbox(
            value=False,
            description='Доп. графики',
            disabled=False,
            indent=False,
            layout=widgets.Layout(width='15%')
        )

        self._cnt_combo.observe(lambda _: self._update_cnt_combo(), names='value')
        self._min_iv.observe(lambda _: self._update_min_iv(), names='value')
        self._min_bucket.observe(lambda _: self._update_min_bucket(), names='value')
        self._segment.observe(lambda _: self._update_segment(), names='value')
        self._with_dop_plots.observe(lambda _: self._update_with_dop_plots(), names='value')
        self._calc_button.on_click(lambda _: self._calc_segments())

        display(widgets.HBox([self._cnt_combo, self._min_bucket, self._min_iv, self._target]))
        display(widgets.HBox([self._segment, self._calc_button]))
        display(widgets.HBox([self._with_dop_plots, self._plot_output]))
        display(self._table_output)

    def _update_cnt_combo(self):
        cnt_combo_min, cnt_combo_max = self._cnt_combo.value
        basic_key = (self._min_bucket.value, cnt_combo_min, cnt_combo_max, self._target.value)
        exist_key = _check_supper_key(basic_key, list(self._cache_report.keys()))

        if exist_key:
            segments_info = self._cache_report[exist_key]['info']
            segments_info = [
                (seg.name, seg) for seg in segments_info
                if cnt_combo_min <= seg.cnt_vars <= cnt_combo_max and seg.iv >= self._min_iv.value
            ]
            self._segment.options = segments_info

            if (self._segment.value is not None
                    and (self._segment.value.name, self._segment.value) not in self._segment.options):
                self._segment.value = segments_info[0][1]
        else:
            self._segment.options = [NONE_VALUE]
            self._segment.value = NONE_VALUE
            self._calc_button.disabled = False

    def _update_min_iv(self):
        cnt_combo_min, cnt_combo_max = self._cnt_combo.value
        basic_key = (self._min_bucket.value, cnt_combo_min, cnt_combo_max, self._target.value)
        exist_key = _check_supper_key(basic_key, list(self._cache_report.keys()))

        if exist_key:
            segments_info = self._cache_report[exist_key]['info']
            segments_info = [
                (seg.name, seg) for seg in segments_info
                if cnt_combo_min <= seg.cnt_vars <= cnt_combo_max and seg.iv >= self._min_iv.value
            ]
            self._segment.options = segments_info
            if self._segment.value is not None and self._segment.value.iv <= self._min_iv.value:
                self._segment.value = segments_info[0][1]

    def _update_min_bucket(self):
        cnt_combo_min, cnt_combo_max = self._cnt_combo.value
        basic_key = (self._min_bucket.value, cnt_combo_min, cnt_combo_max, self._target.value)
        exist_key = _check_supper_key(basic_key, list(self._cache_report.keys()))

        if exist_key:
            segments_info = self._cache_report[exist_key]['info']
            segments_info = [
                (seg.name, seg) for seg in segments_info
                if cnt_combo_min <= seg.cnt_vars <= cnt_combo_max and seg.iv >= self._min_iv.value
            ]
            self._segment.options = segments_info

            if (self._segment.value is not None
                    and (self._segment.value.name, self._segment.value) not in self._segment.options):
                self._segment.value = segments_info[0][1]
        else:
            self._segment.options = [NONE_VALUE]
            self._segment.value = NONE_VALUE
            self._calc_button.disabled = False

    def _calc_segments(self):
        cnt_combo_min, cnt_combo_max = self._cnt_combo.value
        basic_key = (self._min_bucket.value, cnt_combo_min, cnt_combo_max, self._target.value)
        exist_key = _check_supper_key(basic_key, list(self._cache_report.keys()))

        if exist_key:
            if basic_key == exist_key:
                # List[(segment_name, iv, cnt_vars, ((var_name, var_value)))
                segments_info = self._cache_report[exist_key]['info']

            else:
                segments_info = self._cache_report[exist_key]['info']
                segments_info = [s for s in segments_info if cnt_combo_min <= s.cnt_vars <= cnt_combo_max]
        else:
            sub_key = _check_sub_key(basic_key, list(self._cache_report.keys()))
            if sub_key:
                del self._cache_report[sub_key]

            report = calc_concentration_report(
                self._origin_df,
                target_name=self._target.value,
                combo_min=cnt_combo_min,
                combo_max=cnt_combo_max,
                analyze_vars=self._var_columns,
                binning=BinningParams(min_prc=self._min_bucket.value),
                _logging=False
            )
            if len(self._cache_report) == self._max_cached_reports:
                first_key = list(self._cache_report.keys())[0]
                del self._cache_report[first_key]

            order_index = report[C_GROUP_IV.n].abs().sort_values(ascending=False).index
            report = report.loc[order_index].reset_index(drop=True)
            segments_info = report.apply(_extract_segment_info, axis=1).to_list()
            self._cache_report[basic_key] = {'report': report, 'info': segments_info}

        segments_info = [(seg.name, seg) for seg in segments_info if seg.iv >= self._min_iv.value]
        self._segment.options = segments_info
        self._segment.value = segments_info[0][1]

        self._calc_button.disabled = True

    def _update_segment(self):
        segment = self._segment.value
        if segment == NONE_VALUE:
            with self._plot_output:
                self._plot_output.clear_output()

            with self._table_output:
                self._table_output.clear_output()
            return

        select_cols = [var_name for var_name, _ in segment.vars_vals]
        cnt_vars = len(select_cols)

        sub_df = get_sub_df(self._origin_df, select_cols + [self._target.value])
        sub_df = preprocess_df(
            sub_df, select_cols, None, self._target.value,
            BinningParams(min_prc=self._min_bucket.value),
            None, _validate_target=False, drop_not_processed=False,
            _bin_by_target=True, _copy=False
        )
        total_cnt = sub_df.shape[0]
        filtered_sub_df = _filter_df(sub_df, segment.vars_vals)
        stats_dict = _prepare_stats(filtered_sub_df, segment.vars_vals, self._target.value, self._origin_df.shape[0])

        with self._plot_output:
            self._plot_output.clear_output()
            if cnt_vars == 1:
                pass # var_plot
            if cnt_vars in [2, 3]:
                _venn_diagram(stats_dict, segment.vars_vals, total_cnt)
            else:
                pass # another plot

            if self._with_dop_plots.value or cnt_vars == 1:
                for var_name, var_value in segment.vars_vals:
                    plot_config = PlotConfig(plot_size=(8, 4), title=var_name)
                    _ = plot_var_stat(
                        sub_df, var_name, self._target.value,
                        binning=False,
                        plot_config=plot_config,
                        _mark_bar=var_value
                    )
            plt.show()

        with self._table_output:
            self._table_output.clear_output()
            cnt_combo_min, cnt_combo_max = self._cnt_combo.value
            basic_key = (self._min_bucket.value, cnt_combo_min, cnt_combo_max, self._target.value)
            exist_key = _check_supper_key(basic_key, list(self._cache_report.keys()))

            report = self._cache_report[exist_key]['report']
            sub_report = _filter_report(report, segment.vars_vals)
            display(HTML(sub_report.to_html()))

    def _update_with_dop_plots(self):
        self._update_segment()


def _filter_report(report, vars_vals) -> DataFrame:
    max_cnt_cols = len(vars_vals)
    report_columns = get_columns(report)
    full_indx = []
    drop_columns = []
    sort_columns = []

    for i in range(100):
        var_name_col, var_val_col = f"Var{i+1}_Name", f"Var{i+1}_Value"
        if var_name_col not in report_columns:
            break
        elif i+1 > max_cnt_cols:
            sub_indx = report[var_name_col].isnull()
            full_indx.append(sub_indx)
            drop_columns.extend([var_name_col, var_val_col])
        else:
            sub_indx = []
            for var, val in vars_vals:
                indx = (report[var_name_col] == var) & (report[var_val_col] == val)
                sub_indx.append(indx)
            sub_indx.append(report[var_name_col].isnull())
            sub_indx = reduce(lambda x, y: x | y, sub_indx)
            full_indx.append(sub_indx)
            sort_columns.append(var_name_col)

    full_indx = reduce(lambda x, y: x & y, full_indx)
    report = report[full_indx]
    if drop_columns:
        report = report.drop(columns=drop_columns)
    report = report.sort_values(sort_columns[::-1]).reset_index(drop=True)
    return report


def _filter_df(df, vars_vals):
    framework = get_framework_from_dataframe(df)
    func = _MAP_FRAMEWORK_filter_df[framework]
    return func(df, vars_vals)


def _filter_df_polars(df, vars_vals):
    import polars as pl
    df = df.filter(reduce(
        lambda x, y: x | y,
        [pl.col(var_name) == var_value for var_name, var_value in vars_vals]
    ))
    return df


def _filter_df_pandas(df, vars_vals):
    select_indxs = []
    for var_name, var_value in vars_vals:
        indx = df[var_name] == var_value
        select_indxs.append(indx)

    select_indxs = reduce(lambda x, y: x | y, select_indxs)
    df = df.loc[select_indxs].reset_index(drop=True)
    return df


_MAP_FRAMEWORK_filter_df = {
    FrameWork.pandas: _filter_df_pandas,
    FrameWork.polars: _filter_df_polars,
}


def _prepare_stats(df, vars_vals, target_name, cnt_total_obs):
    import polars as pl
    cnt_vars = len(vars_vals)

    main_filters = {name: pl.col(name) == val for name, val in vars_vals}

    all_combos = [['0'], ['1']]
    for i in range(cnt_vars - 1):
        new_combos = []
        for combo in all_combos:
            for i in range(2):
                new_combos.append(combo + [str(i),])
        all_combos = new_combos

    all_combos = all_combos[1:]

    framework = get_framework_from_dataframe(df)
    if framework is FrameWork.polars:
        stats_dict = _prepare_stats_polars(df, vars_vals, target_name, cnt_total_obs, all_combos)
    elif framework is FrameWork.pandas:
        stats_dict = _prepare_stats_pandas(df, vars_vals, target_name, cnt_total_obs, all_combos)
    else:
        raise NotImplementedError
    return stats_dict


def _prepare_stats_polars(df, vars_vals, target_name, cnt_total_obs, all_combos):
    import polars as pl
    main_filters = {name: pl.col(name) == val for name, val in vars_vals}
    stats_dict = dict()

    for combo in all_combos:
        indx = [
            main_filters[vars_vals[i][0]] if int(equal) else ~main_filters[vars_vals[i][0]]
            for i, equal in enumerate(combo)
        ]
        indx = reduce(lambda x, y: x & y, indx)
        combo_df = df.filter(indx)
        cnt_obs = combo_df.shape[0]
        cnt_target = combo_df[target_name].sum()
        target_rate = 100 * (cnt_target or 1) / cnt_obs
        stats_dict[''.join(combo)] = [cnt_obs, 100 * cnt_obs / cnt_total_obs, cnt_target, target_rate]

    return stats_dict


def _prepare_stats_pandas(df, vars_vals, target_name, cnt_total_obs, all_combos):
    main_filters = {name: df[name] == val for name, val in vars_vals}
    stats_dict = dict()

    for combo in all_combos:
        indx = [
            main_filters[vars_vals[i][0]] if int(equal) else ~main_filters[vars_vals[i][0]]
            for i, equal in enumerate(combo)
        ]
        indx = reduce(lambda x, y: x & y, indx)
        combo_df = df.loc[indx]
        cnt_obs = combo_df.shape[0]
        cnt_target = combo_df[target_name].sum()
        target_rate = 100 * (cnt_target or 1) / cnt_obs
        stats_dict[''.join(combo)] = [cnt_obs, 100 * cnt_obs / cnt_total_obs, cnt_target, target_rate]

    return stats_dict


def _venn_diagram(stats_dict, vars_vals, total_cnt):
    cnt_vars = len(vars_vals)
    if cnt_vars == 2:
        stats_dict = {key + '0': val for key, val in stats_dict.items()}

    population_data = {seg_id: pop for seg_id, (_, pop, _, _) in stats_dict.items()}
    tr_data = {seg_id: tr for seg_id, (*_, tr) in stats_dict.items()}

    total_stats = {
        i: [(cnt_obs, cnt_tr) for seg_id, (cnt_obs, _, cnt_tr, _)  in stats_dict.items() if seg_id[i] == '1']
        for i, _ in enumerate(vars_vals)
    }
    total_pop = {
        i: 100 * sum([cnt_obs for cnt_obs, _ in v]) / total_cnt
        for i, v in total_stats.items()
    }
    total_tr = {
        i: 100 * sum([cnt_tr for _, cnt_tr in v]) / sum([cnt_obs for cnt_obs, _ in v])
        for i, v in total_stats.items()
    }

    labels = [
        f'{var_name} = "{var_value}"\nPop-n = {round(total_pop[i], 1)}%\nTargetRate = {round(total_tr[i], 1)}%'
        for i, (var_name, var_value) in enumerate(vars_vals)
    ]

    if cnt_vars == 2:
        labels.append('')
    plot_config = PlotConfig(plot_size=(8, 8))
    fig = plt.figure(figsize=plot_config.plot_size)
    ax = fig.add_subplot(111)
    venn_diag = venn3(population_data, set_labels=tuple(labels), ax=ax)
    palette = sns.color_palette('Blues', 121).as_hex()
    max_tr, min_tr = max(tr_data.values()), min(tr_data.values())

    for seg_id, tr in tr_data.items():
        color_ind = int(100 * (tr - min_tr) / (max_tr - min_tr))
        venn_diag.get_patch_by_id(seg_id).set_color(palette[20+color_ind])
        venn_diag.get_label_by_id(seg_id).set_text(f"{round(tr, 1)}%")


def _check_supper_key(key, other_keys: List):
    min_prc, cnt_combo_min, cnt_combo_max, target = key
    for gen_combo_min in range(1, cnt_combo_min + 1):
        for gen_combo_max in range(cnt_combo_max, MAX_COMBO + 1):
            gen_key = (min_prc, gen_combo_min, gen_combo_max, target)
            if gen_key in other_keys:
                return gen_key


def _check_sub_key(key, other_keys: List):
    min_prc, cnt_combo_min, cnt_combo_max, target = key
    for existed_key in other_keys:
        ex_min_prc, ex_cnt_combo_min, ex_cnt_combo_max, ex_target = existed_key
        if min_prc == ex_min_prc and ex_target == target:
            if cnt_combo_min <= ex_cnt_combo_min and cnt_combo_max >= ex_cnt_combo_max:
                return existed_key


def _extract_segment_info(ser):
    import pandas as pd
    iv = str(round(ser[C_GROUP_IV.n], 1))
    vars_vals_str = []
    vars_vals = []
    cnt_vars = 0
    for i in range(1, MAX_COMBO + 1):
        var_name = f'Var{i}_Name'
        var_value = f'Var{i}_Value'

        if var_name not in ser.index:
            break
        elif pd.isnull(ser[var_value]):
            break

        vars_vals_str.append(f'({ser[var_name]} = "{ser[var_value]}")')
        vars_vals.append((ser[var_name], ser[var_value]))
        cnt_vars += 1

    vars_vals_str = ' & '.join(vars_vals_str)
    segment_info = SegmentInfo(
        name=f'({iv}) {vars_vals_str}',
        iv=abs(ser[C_GROUP_IV.n]),
        cnt_vars=cnt_vars,
        vars_vals=tuple(vars_vals)
    )
    return segment_info


@dataclass
class SegmentInfo:
    __slots__ = ('name', 'iv', 'cnt_vars', 'vars_vals')
    name: str
    iv: float
    cnt_vars: int
    vars_vals: tuple
