from abc import ABC, abstractmethod

import numpy as np

from typing import List, Dict, Tuple
from pandas.core.frame import DataFrame
from pandas.core.index import Index


class Backtesting(ABC):

    def __init__(self, data: DataFrame, assets: List[str]) -> None:
        self._index, self._data = self._handle_data(data, assets)
        self.assets = assets

    @abstractmethod
    def logic(self, counter: int, prices: np.ndarray) -> float:
        pass

    def apply_strategy(self) -> List[float]:
        return [self.logic(i, row) for i, row in enumerate(self._data)]

    @staticmethod
    def _handle_data(data: DataFrame, assets: List[str]) -> Tuple[Index, np.ndarray]:
        columns: List[str] = list(set(c.split('_')[-1] for c in data.columns))
        to_drop: List[str] = [c for c in columns if c not in assets]
        for c in to_drop:
            data = data.drop(list(data.filter(like=c).columns), axis=1)
        return data.index, data.to_numpy()
