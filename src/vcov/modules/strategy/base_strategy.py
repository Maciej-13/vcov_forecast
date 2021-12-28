import numpy as np

from abc import ABC, abstractmethod
from typing import List, Tuple, Union, Optional
from pandas.core.frame import DataFrame
from pandas import Index, Series

from vcov.modules.trade.trade import TradeHistory


class Strategy(ABC):

    def __init__(self, data: DataFrame, portfolio_value: Union[int, float], fee_multiplier: Optional[float]) -> None:
        self._index, self._data = self._handle_data(data)
        self.assets: List[str] = list(data.columns)
        self.portfolio_value = portfolio_value
        self.trading = TradeHistory()
        self.fee_multiplier: Optional[float] = fee_multiplier

    @abstractmethod
    def logic(self, counter: int, prices: np.ndarray) -> float:
        raise NotImplementedError("Abstract method must be implemented in the derived class!")

    def apply_strategy(self) -> Series:
        return Series([self.logic(i, row) for i, row in enumerate(self._data)], index=self._index)

    def _get_slice(self, current_observation: int, last_observations: int) -> np.ndarray:
        start: int = current_observation - last_observations + 1
        start = start if start >= 0 else 0
        return self._data[start:current_observation + 1, :]

    @staticmethod
    def _handle_data(data: DataFrame) -> Tuple[Index, np.ndarray]:
        return data.index, data.to_numpy()
