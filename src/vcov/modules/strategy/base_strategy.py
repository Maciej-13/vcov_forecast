import numpy as np

from abc import ABC, abstractmethod
from typing import List, Union, Optional
from pandas.core.frame import DataFrame
from pandas import Series, Timestamp

from vcov.modules.trade.trade import TradeHistory
from vcov.modules.portfolio.portfolio import Portfolio


class Strategy(ABC):

    def __init__(self, data: DataFrame, portfolio_value: Union[int, float], fee_multiplier: Optional[float]) -> None:
        self._data = data
        self.assets: List[str] = list(data.columns)
        self.portfolio_value = portfolio_value
        self.portfolio = Portfolio(assets=self.assets)
        self.trading = TradeHistory()
        self.fee_multiplier: Optional[float] = fee_multiplier

    @abstractmethod
    def logic(self, counter: int, prices: Series, **kwargs) -> Optional[Union[float, np.ndarray]]:
        raise NotImplementedError("Abstract method must be implemented in the derived class!")

    def apply_strategy(self, **kwargs) -> Series:
        return Series([self.logic(i, self._data.iloc[i], **kwargs) for i in range(len(self._data))],
                      index=self._data.index)

    def _get_slice(self, current_idx: Timestamp, last_observations: int) -> DataFrame:
        df = self._data.loc[self._data.index <= current_idx].copy(deep=True)
        return df.tail(last_observations).copy(deep=True)
