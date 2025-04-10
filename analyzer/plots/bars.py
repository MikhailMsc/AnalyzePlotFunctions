from analyzer import calc_iv_var, calc_var_groups_stat
from preprocessing.preprocess import BinningParamsSingleVars, MapDictSingleVar
from utils.general.types import DataFrame


def plot_var_stat(
        data: DataFrame, var_name: str, target_name: str = None,
        map_values: MapDictSingleVar = None, binning: BinningParamsSingleVars = True,
        del_values: list = None, xticks='name', annotation: bool = True, plot_config=None,
        print_points=True, return_table=True
):
    """
    План решения:
    1. Нужно как-то бинаризовать переменную, если это необходимо.
    2. В зависимости есть ли target или нет, получить таблицу с необходимой статистикой.
    3. Нарисовать график
    4. Отобразить табличку
    """

    if target_name is not None:
        report = calc_iv_var(data, var_name, target_name, binning, map_values)
    else:
        report = calc_var_groups_stat(data, var_name, binning, map_values)
