import numpy as np
import pandas as pd

from pandas import DataFrame
from pypfopt.discrete_allocation import DiscreteAllocation
from typing import List, Union, Optional, Dict

from vcov.modules.strategy.base_strategy import Strategy
from vcov.modules.portfolio.portfolio import Portfolio


def resolve_allocation(assets: List[str], weights: Dict[str, float], prices: np.ndarray,
                       portfolio_value: Union[int, float]) -> List[int]:
    allocation = DiscreteAllocation(weights, pd.Series(prices, index=assets),
                                    portfolio_value).greedy_portfolio()[0]
    return [allocation.get(i) if allocation.get(i) is not None else 0 for i in assets]


class EquallyWeighted(Strategy):

    def __init__(self, data: DataFrame, portfolio_value: Union[int, float], fee_multiplier: Optional[float]) -> None:
        super().__init__(data, portfolio_value, fee_multiplier)
        self.portfolio = Portfolio(assets=self.assets)

    def logic(self, counter: int, prices: np.ndarray) -> Union[float, np.ndarray]:
        if counter == 0:
            self.portfolio.update_weights(
                {i: 1 / len(self.assets) for i in self.assets}
            )
            self.trading.register(
                stamp=counter,
                asset=self.assets,
                quantity=resolve_allocation(self.assets, self.portfolio.weights, prices, self.portfolio_value),
                price=prices,
                buy=True,
                fee_multiplier=self.fee_multiplier
            )
        return np.dot(np.fromiter(self.portfolio.weights.values(), dtype=float), prices)
