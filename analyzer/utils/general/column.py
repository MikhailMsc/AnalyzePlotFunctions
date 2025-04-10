class Column:

    def __init__(self, name: str):
        self._name = name

    @property
    def n(self) -> str:
        return self._name

    def child(self, n) -> 'Column':
        return self.__class__(n)