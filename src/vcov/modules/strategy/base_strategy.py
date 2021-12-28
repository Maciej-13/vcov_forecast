import numpy as np

from abc import ABC, abstractmethod
from typing import List, Tuple, Union, Optional
from pandas.core.frame import DataFrame
from pandas import Index

from vcov.modules.trade.trade import TradeHistory


class Strategy(ABC):

    def __init__(self, data: DataFrame, assets: List[str], portfolio_value: Union[int, float],
                 fee_multiplier: Optional[float]) -> None:
        self._index, self._data = self._handle_data(data, assets)
        self.assets = assets
        self.portfolio_value = portfolio_value
        self.trading = TradeHistory()
        self.fee_multiplier: Optional[float] = fee_multiplier

    @abstractmethod
    def logic(self, counter: int, prices: np.ndarray) -> float:
        raise NotImplementedError("Abstract method must be implemented in the derived class!")

    def apply_strategy(self) -> List[float]:
        return [self.logic(i, row) for i, row in enumerate(self._data)]

    def _get_slice(self, current_observation: int, last_observations: int) -> np.ndarray:
        start: int = current_observation - last_observations + 1
        start = start if start >= 0 else 0
        return self._data[start:current_observation + 1, :]

    @staticmethod
    def _handle_data(data: DataFrame, assets: List[str]) -> Tuple[Index, np.ndarray]:
        columns: List[str] = list(set(c.split('_')[-1] for c in data.columns))
        to_drop: List[str] = [c for c in columns if c not in assets]
        for c in to_drop:
            data = data.drop(list(data.filter(like=c).columns), axis=1)
        return data.index, data.to_numpy()
