from typing import List

from logger import WithLogger

from ..framework_depends.columns import get_columns, drop_columns
from .column import Column
from .types import DataFrame


class SchemaDF(WithLogger):
    _log_msg_template = 'Схема данных'

    def __init__(self, columns: List[Column], key: List[Column] = None, name: str = None):
        if not columns:
            raise ValueError('Схема обязана иметь список колонок!')
        self._columns = columns
        self._key = key or []
        self._name = name or ''

        self._col_names = None
        self._key_names = None

    @property
    def t(self) -> DataFrame:
        return DataFrame

    def _get_log_template(self) -> str:
        if self._name:
            return self._log_msg_template + f' ({self._name})'
        else:
            return self._log_msg_template

    @property
    def col_names(self) -> List[str]:
        if self._col_names is None:
            self._col_names = [col.n for col in self._columns]
        return self._col_names

    @property
    def key_names(self) -> List[str]:
        if self._key_names is None:
            self._key_names = [col.name for col in self._key]
        return self._key_names

    def __call__(self, df: DataFrame):
        df = self._validate_columns(df)
        return df

    def _validate_columns(self, df: DataFrame):
        observe_columns = set(get_columns(df))
        schema_cols = set(self.col_names)
        need = schema_cols - observe_columns
        not_need = observe_columns - schema_cols

        if need:
            exception_msg = f'Валидация | Отсутствуют необходимые колонки - {need}'
            raise Exception(exception_msg)

        if not_need:
            df = drop_columns(df, list(not_need))

        return df



