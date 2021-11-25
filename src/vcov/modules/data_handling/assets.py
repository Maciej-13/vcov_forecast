from __future__ import annotations
import pandas as pd
import numpy as np

from typing import List, Tuple
from pandas.core.frame import DataFrame


class Assets:

    def __init__(self, names: List[str], prices: DataFrame, log_returns: bool = False):
        self._names = names
        self._prices = prices
        self._returns: DataFrame = self._calculate_returns(log_returns)

    def get_names(self) -> List[str]:
        return self._names

    def get_prices(self) -> DataFrame:
        return self._prices

    def get_returns(self) -> DataFrame:
        return self._returns

    def split_by_date(self, date: str) -> Tuple[Assets, Assets]:
        date = pd.to_datetime(date)
        in_sample = self._prices.loc[self._prices.index < date].copy(deep=True)
        out_of_sample = self._prices.loc[self._prices.index >= date].copy(deep=True)
        return Assets(names=self._names, prices=in_sample), Assets(names=self._names, prices=out_of_sample)

    def split_by_length(self, split: float = 0.8) -> Tuple[Assets, Assets]:
        index: int = int(len(self._prices.index) * split) - 1
        index = 0 if index < 0 else index
        in_sample = self._prices.loc[self._prices.index <= self._prices.index[index]].copy(deep=True)
        out_of_sample = self._prices.loc[self._prices.index > self._prices.index[index]].copy(deep=True)
        return Assets(names=self._names, prices=in_sample), Assets(names=self._names, prices=out_of_sample)

    def _calculate_returns(self, log: bool = False) -> DataFrame:
        if log:
            return np.log1p(self._prices.pct_change())
        return self._prices.pct_change()
