from __future__ import annotations
import pandas as pd
import numpy as np

from dataclasses import dataclass
from typing import List, Tuple, Optional
from pandas.core.frame import DataFrame


@dataclass
class Assets:
    prices: DataFrame
    tickers: Optional[List[str]] = None
    log_returns: bool = False

    def __post_init__(self):
        self.tickers = [c.split("_")[-1] for c in self.prices.columns]
        self.prices.columns = self.tickers
        self.returns: DataFrame = self._calculate_returns(self.log_returns)

    def split_by_date(self, date: str) -> Tuple[Assets, Assets]:
        date = pd.to_datetime(date)
        in_sample = self.prices.loc[self.prices.index < date].copy(deep=True)
        out_of_sample = self.prices.loc[self.prices.index >= date].copy(deep=True)
        return Assets(tickers=self.tickers, prices=in_sample), Assets(tickers=self.tickers, prices=out_of_sample)

    def split_by_length(self, split: float = 0.8) -> Tuple[Assets, Assets]:
        index: int = int(len(self.prices.index) * split) - 1
        index = 0 if index < 0 else index
        in_sample = self.prices.loc[self.prices.index <= self.prices.index[index]].copy(deep=True)
        out_of_sample = self.prices.loc[self.prices.index > self.prices.index[index]].copy(deep=True)
        return Assets(tickers=self.tickers, prices=in_sample), Assets(tickers=self.tickers, prices=out_of_sample)

    def _calculate_returns(self, log: bool = False) -> DataFrame:
        if log:
            return np.log1p(self.prices.pct_change())
        return self.prices.pct_change()
