import numpy as np

from abc import ABC, abstractmethod
from typing import List, Union, Optional, Dict

import pandas as pd
from pandas.core.frame import DataFrame
from pandas import Series, Timestamp

from vcov.modules.trade.trade import TradeHistory
from vcov.modules.portfolio.portfolio import Portfolio


def find_last_available_date(collection: Dict[str, List[str]], date: Timestamp) -> List[str]:
    collection = {pd.to_datetime(k): v for k, v in collection.items()}
    if date in collection.keys():
        return collection[date]
    else:
        for dt in pd.date_range(next(iter(collection.keys())), date, freq='D')[::-1]:
            try:
                return collection[dt]
            except KeyError:
                pass


class Strategy(ABC):

    def __init__(self, data: DataFrame, portfolio_value: Union[int, float], fee_multiplier: Optional[float],
                 save_results: str, market_cap_selection: Optional[Dict[str, List[str]]] = None) -> None:
        self._data = data
        self.assets: List[str] = list(data.columns)
        self.portfolio_value = portfolio_value
        self.cash: float = 0.0
        self.portfolio = Portfolio(assets=self.assets)
        self.trading = TradeHistory()
        self.fee_multiplier: Optional[float] = fee_multiplier
        self._path: str = save_results
        self._selection: Dict[str, List[str]] = market_cap_selection

    @abstractmethod
    def logic(self, counter: int, prices: Series, **kwargs) -> Optional[Union[float, np.ndarray]]:
        raise NotImplementedError("Abstract method must be implemented in the derived class!")

    def apply_strategy(self, **kwargs) -> Series:
        return Series([self.logic(i, self._data.iloc[i], **kwargs) for i in range(len(self._data))],
                      index=self._data.index)

    def _calculate_portfolio_value(self, prices) -> float:
        portfolio_value: float = self.cash - self.trading.accumulated_fees + np.dot(
            np.fromiter(self.portfolio.stocks.values(), dtype=float), prices[self.portfolio.stocks.keys()])
        if portfolio_value <= 0:
            return 0.0
        return portfolio_value

    def _get_slice(self, current_idx: Timestamp, last_observations: Optional[int]) -> DataFrame:
        if self._selection is None:
            assets = self.assets
        else:
            assets = find_last_available_date(self._selection, current_idx)
        df = self._data.loc[self._data.index <= current_idx, assets].copy(deep=True)
        if last_observations is not None:
            return df.tail(last_observations).copy(deep=True)
        else:
            return df.copy(deep=True)
