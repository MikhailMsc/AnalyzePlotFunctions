from typing import List, Dict

from analyzer.logger import WithLogger

from ..framework_depends.columns import get_columns, drop_columns, reorder_columns
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

    @property
    def col_names(self) -> List[str]:
        if self._col_names is None:
            self._col_names = [col.n for col in self._columns]
        return self._col_names

    @property
    def key_names(self) -> List[str]:
        if self._key_names is None:
            self._key_names = [col.n for col in self._key]
        return self._key_names

    def copy(self) -> 'SchemaDF':
        sh_copy = self.__class__(columns=self._columns[:], key=self._key[:], name=self._name)
        return sh_copy

    def reorder_columns(self, df: DataFrame) -> DataFrame:
        observe_columns = set(get_columns(df))
        schema_cols = set(self.col_names)

        if observe_columns != schema_cols:
            need_columns = schema_cols - observe_columns
            not_need_columns = observe_columns - schema_cols

            exception_msg = self._get_log_template()
            exception_msg += f' | reorder_columns | {need_columns = }, {not_need_columns = }, '
            raise Exception(exception_msg)
        else:
            df = reorder_columns(df, self.col_names)

        return df

    def replace_columns(self, replace_dict: Dict[str, Column]):
        for pos, col_name in enumerate(self.col_names):
            if col_name in replace_dict:
                self._columns[pos] = replace_dict[col_name]
        self._col_names = None

        for pos, col_name in enumerate(self.key_names):
            if col_name in replace_dict:
                self._key[pos] = replace_dict[col_name]
        self._key_names = None

    def delete_columns(self, columns: list[Column]):
        for col in columns:
            if col in self._columns:
                self._columns.remove(col)
            if col in self._key:
                self._key.remove(col)

    def add_columns(self, cols_dict: Dict[Column, bool], pos: int):
        pos = min(len(self._columns), pos)
        if pos == -1:
            self._columns = self._columns + list(cols_dict.keys())
        else:
            self._columns = self._columns[:pos] + list(cols_dict.keys()) + self._columns[pos:]

        new_keys = [col for col, is_key in cols_dict.items() if is_key]
        self._key += new_keys

    def __call__(self, df: DataFrame, check_duplicates: bool = False, reorder_colums: bool = False):
        # TODO: check_duplicates добавил на перспективу
        df = self._validate_columns(df)

        if reorder_colums:
            df = self.reorder_columns(df)
        return df

    def _validate_columns(self, df: DataFrame):
        observe_columns = set(get_columns(df))
        schema_cols = set(self.col_names)
        need = schema_cols - observe_columns
        not_need = observe_columns - schema_cols

        if need:
            exception_msg = self._get_log_template()
            exception_msg += f' | Валидация | Отсутствуют необходимые колонки - {need}'
            raise Exception(exception_msg)

        if not_need:
            df = drop_columns(df, list(not_need))

        return df

    def _get_log_template(self) -> str:
        if self._name:
            return self._log_msg_template + f' ({self._name})'
        else:
            return self._log_msg_template



