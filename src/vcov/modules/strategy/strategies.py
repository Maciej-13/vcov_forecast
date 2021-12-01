import numpy as np

from pandas import DataFrame
from typing import List, Union

from vcov.modules.strategy.base_strategy import Strategy
from vcov.modules.strategy.portfolio import Portfolio


class EquallyWeighted(Strategy):

    def __init__(self, data: DataFrame, assets: List[str]) -> None:
        super().__init__(data, assets)
        self.portfolio = Portfolio(assets=assets)

    def logic(self, counter: int, prices: np.ndarray) -> Union[float, np.ndarray]:
        if counter == 0:
            self.portfolio.update_weights(
                {i: 1 / len(self.assets) for i in self.assets}
            )
            self.portfolio.update_buy_prices(
                {i: prices[j] for j, i in enumerate(self.assets)}
            )
        return np.dot(np.fromiter(self.portfolio.weights.values(), dtype=float), prices)
