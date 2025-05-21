from typing import Union

from analyzer.preprocessing import BinningParams, MapDictSingleVar, preprocess_df
from analyzer.utils.domain.columns import C_GROUP_NUMBER, C_GROUP, C_TOTAL, C_POPULATION
from analyzer.utils.framework_depends import convert_df_to_pandas
from analyzer.utils.general.schema import SchemaDF
from analyzer.utils.general.types import DataFrame, get_framework_from_dataframe, FrameWork

SH_GroupsStatReport = SchemaDF(
    columns=[
        C_GROUP_NUMBER, C_GROUP, C_TOTAL, C_POPULATION,
    ],
    key=[C_GROUP_NUMBER]
)


def calc_var_groups_stat(
        df: DataFrame, var_name: str, map_values: MapDictSingleVar = None,
        binning: Union[BinningParams, bool] = True
) -> SH_GroupsStatReport.t:
    """
    Общая статистика по размеру популяций групп одной переменной.

    Args:
        df:         Датафрейм, содержащий интересующую переменную.
        var_name:   Название интересующей переменной.
        map_values: Словарь для мэппинга значений переменной.
        binning:    Параметры для биннинга.

    Returns:
        DataFrame:
            GROUP_NUMBER:   Номер группы
            GROUP:          Название группы
            TOTAL:          Размер группы, кол-во наблюдений
            POPULATION:     Размер группы, процент выборки
    """
    if binning or map_values:
        if map_values is not None:
            map_values = {var_name: map_values}

        df = preprocess_df(
            df, [var_name], map_values=map_values, binning=binning,
            drop_not_processed=True, _tqdm=False
        )

    framework = get_framework_from_dataframe(df)
    func = _MAP_FRAMEWORK_FUNC[framework]
    groups_report = func(df, var_name)

    groups_report[C_GROUP_NUMBER.n] = groups_report.index
    groups_report[C_GROUP_NUMBER.n] += 1

    groups_report = SH_GroupsStatReport(groups_report, reorder_colums=True)
    return groups_report


def _calc_var_groups_stat_pd(df: DataFrame, var_name: str):
    df = df.groupby(var_name, as_index=False, observed=True).size()
    df.columns = [C_GROUP.n, C_TOTAL.n]

    total_sum = df[C_TOTAL.n].sum()
    df[C_POPULATION.n] = round(100 * df[C_TOTAL.n] / total_sum, 2)
    return df


def _calc_var_groups_stat_pl(df: DataFrame, var_name: str):
    import polars as pl

    df = df.group_by(var_name).len().rename({var_name: C_GROUP.n, 'len': C_TOTAL.n})
    total_sum = df[C_TOTAL.n].sum()
    df = df.with_columns(
        (100 * pl.col(C_TOTAL.n) / total_sum).round(2).alias(C_POPULATION.n)
    )
    df = df.sort(C_GROUP.n)
    df = convert_df_to_pandas(df)
    return df


def _calc_var_groups_stat_spark(df: DataFrame, var_name: str):
    raise NotImplementedError


_MAP_FRAMEWORK_FUNC = {
    FrameWork.pandas: _calc_var_groups_stat_pd,
    FrameWork.polars: _calc_var_groups_stat_pl,
    FrameWork.spark: _calc_var_groups_stat_spark,

}