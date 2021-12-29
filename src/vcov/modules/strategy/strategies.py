import inspect
import numpy as np

from pandas import DataFrame, Series
from typing import List, Union, Optional, Dict

from pypfopt.discrete_allocation import DiscreteAllocation
from pypfopt import risk_models, expected_returns, EfficientFrontier
from pypfopt.objective_functions import transaction_cost

from vcov.modules.strategy.base_strategy import Strategy


def resolve_allocation(weights: Dict[str, float], prices: Series, portfolio_value: Union[int, float]) -> List[int]:
    allocation = DiscreteAllocation(weights, prices, portfolio_value).greedy_portfolio()[0]
    return [allocation.get(i) if allocation.get(i) is not None else 0 for i in prices.index]


class EquallyWeighted(Strategy):

    def logic(self, counter: int, prices: Series, **kwargs) -> Union[float, np.ndarray]:
        if counter == 0:
            self.portfolio.update_weights(
                {i: 1 / len(self.assets) for i in self.assets}
            )
            self.trading.register(
                stamp=prices.name,
                asset=self.assets,
                quantity=resolve_allocation(self.portfolio.weights, prices, self.portfolio_value),
                price=prices,
                buy=True,
                fee_multiplier=self.fee_multiplier
            )
        return np.dot(np.fromiter(self.portfolio.weights.values(), dtype=float), prices.values)


class RiskModels(Strategy):

    def __init__(self, data: DataFrame, portfolio_value: Union[int, float], fee_multiplier: Optional[float],
                 window: int, rebalancing: Optional[int]) -> None:
        super().__init__(data, portfolio_value, fee_multiplier)
        self.window = window
        self.rebalancing = rebalancing

    def logic(self, counter: int, prices: Series, **kwargs) -> Optional[Union[float, np.ndarray]]:
        if counter < self.window - 1:
            return None

        if self.rebalancing is None:
            if counter == self.window - 1:
                return self._single_logic(prices, **kwargs)
            else:
                return np.dot(np.fromiter(self.portfolio.weights.values(), dtype=float), prices)

        if self.rebalancing is not None:
            if (counter - (self.window - 1)) % self.rebalancing == 0:
                return self._single_logic(prices, **kwargs)
            else:
                return np.dot(np.fromiter(self.portfolio.weights.values(), dtype=float), prices)

    def _optimize_weights(self, prices: Series, covariance_method: str, returns_method: str, optimize: str,
                          **kwargs) -> None:
        sliced_data = self._get_slice(current_idx=prices.name, last_observations=self.window)
        sample_cov = risk_models.risk_matrix(
            method=covariance_method,
            prices=sliced_data,
            returns_data=False,
            **kwargs
        )
        er = expected_returns.return_model(prices=sliced_data, method=returns_method)
        ef = EfficientFrontier(er, sample_cov, weight_bounds=(0, 1))
        if self.fee_multiplier is not None:
            w_prev = np.fromiter(self.portfolio.weights.values(), dtype=float) if self.portfolio.weights \
                else [0] * len(self.portfolio.assets)
            ef.add_objective(transaction_cost, w_prev=w_prev, k=self.fee_multiplier)
        optimizer = getattr(ef, optimize)
        optimizer(**{k: kwargs.pop(k) for k in kwargs if k in inspect.signature(optimizer).parameters.keys()})
        self.portfolio.update_weights(ef.clean_weights())

    def _single_logic(self, prices: Series, **kwargs) -> Union[float, np.ndarray]:
        self._optimize_weights(prices, **kwargs)
        self.trading.register(
            stamp=prices.name,
            asset=list(self.portfolio.weights.keys()),
            quantity=resolve_allocation(self.portfolio.weights, prices, self.portfolio_value),
            price=prices,
            buy=True,
            fee_multiplier=self.fee_multiplier
        )
        return np.dot(np.fromiter(self.portfolio.weights.values(), dtype=float), prices)
