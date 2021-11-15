import pandas as pd

from sklearn.model_selection import train_test_split
from typing import Optional, Tuple, List
from pandas.core.frame import DataFrame


class InputHandler:

    def __init__(self, path: str, assets: List[str], column: Optional[str] = None, returns: bool = True) -> None:
        self.assets = assets
        self._data = self._load_data(path)
        column = column if column is not None else "Close"
        self._select_column(column)
        if returns:
            self.__calculate_returns()

    def get_data(self) -> DataFrame:
        return self._data

    def train_test_split(self, test_size: float = 0.2) -> Tuple[DataFrame, DataFrame]:
        train, test = train_test_split(self._data, test_size=test_size, shuffle=False, random_state=42)
        return train, test

    def _select_column(self, column: Optional[str]) -> None:
        if column:
            close_col = [column + ' ' + a for a in self.assets]
            self._data = self._data[close_col]
            self._data.columns = [c.split()[-1].strip() for c in self._data.columns]

    def __calculate_returns(self) -> None:
        self._data = self._data.pct_change(1)
        self._data = self._data.dropna(how='all', axis=0)

    @staticmethod
    def _load_data(path: str) -> DataFrame:
        return pd.read_csv(path, parse_dates=['Date'], index_col='Date')
