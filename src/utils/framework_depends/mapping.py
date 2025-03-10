from typing import TypeVar, Dict

from utils.general.types import Series, get_framework_from_series, FrameWork

Source = TypeVar('Source')
Target = TypeVar('Target')


def map_elements(series: Series, map_dict: Dict[Source, Target]) -> Series:
    framework = get_framework_from_series(series)
    func = _MAP_FRAMEWORK_FUNC[framework]
    return func(series, map_dict)


def _map_elements_pandas(series: Series, map_dict: Dict[Source, Target]) -> Series:
    series = series.map(map_dict)
    return series


def _map_elements_polars(series: Series, map_dict: Dict[Source, Target]) -> Series:
    series = series.map_elements(lambda x: map_dict.get(x, x))
    return series


def _map_elements_pyspark(series: Series, map_dict: Dict[Source, Target]) -> Series:
    raise NotImplementedError


_MAP_FRAMEWORK_FUNC = {
    FrameWork.pandas: _map_elements_pandas,
    FrameWork.polars: _map_elements_polars,
    FrameWork.spark: _map_elements_pyspark,
}






