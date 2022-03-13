import inspect
import numpy as np
import pandas as pd
import tensorflow as tf

from pandas import DataFrame, Series
from typing import Union, Optional, Dict, Tuple
from keras import Model

from pypfopt.discrete_allocation import DiscreteAllocation
from pypfopt import risk_models, expected_returns, EfficientFrontier
from pypfopt.objective_functions import transaction_cost

from sklearn.model_selection import train_test_split

from vcov.modules.strategy.base_strategy import Strategy
from vcov.modules.models.lstm_model import LSTM
from vcov.modules.data_handling.covariance_handler import CovarianceHandler
from vcov.modules.data_handling.dataset import KerasDataset

Allocation = Tuple[Dict[str, int], float]
Orders = Tuple[Dict[str, int], Dict[str, int]]


def resolve_allocation(weights: Dict[str, float], prices: Series, portfolio_value: Union[int, float]) -> Allocation:
    allocation, cash = DiscreteAllocation(weights, prices, portfolio_value).greedy_portfolio()
    return {k: (0 if v is None else v) for k, v in allocation.items()}, cash


def resolve_order_amounts(old_stocks: Dict[str, int], new_stocks: Dict[str, int]) -> Orders:
    old_stocks = {k: v for k, v in old_stocks.items() if v != 0}
    new_stocks = {k: v for k, v in new_stocks.items() if v != 0}
    to_sell = {k: v for k, v in old_stocks.items() if k not in new_stocks.keys()}
    to_buy = {k: v for k, v in new_stocks.items() if k not in old_stocks.keys()}
    for k, v in new_stocks.items():
        if k in old_stocks.keys() and v > old_stocks[k]:
            to_buy.update({k: v - old_stocks[k]})
        elif k in old_stocks.keys() and v < old_stocks[k]:
            to_sell.update({k: old_stocks[k] - v})
    return to_sell, to_buy


class EquallyWeighted(Strategy):

    def logic(self, counter: int, prices: Series, **kwargs) -> Union[float, np.ndarray]:
        if counter == 0:
            self.portfolio.update_weights(
                {i: 1 / len(self.assets) for i in self.assets}
            )
            allocation, cash = resolve_allocation(self.portfolio.weights, prices, self.portfolio_value)
            self.cash = cash
            self.portfolio.update_stocks(allocation)
            self.trading.register(
                stamp=prices.name,
                asset=list(allocation.keys()),
                quantity=list(allocation.values()),
                price=prices,
                buy=True,
                fee_multiplier=self.fee_multiplier
            )
            self.portfolio_value = self._calculate_portfolio_value(prices)
        return self._calculate_portfolio_value(prices)


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
                return self._calculate_portfolio_value(prices)

        if self.rebalancing is not None:
            if (counter - (self.window - 1)) % self.rebalancing == 0:
                return self._single_logic(prices, **kwargs)
            else:
                return self._calculate_portfolio_value(prices)

    def _single_logic(self, prices: Series, **kwargs) -> Union[float, np.ndarray]:
        new_weights = self._optimize_weights(prices, **kwargs)
        self.portfolio.update_weights(new_weights)
        allocation, cash = resolve_allocation(self.portfolio.weights, prices, self.portfolio_value)
        self.cash = cash
        to_sell, to_buy = resolve_order_amounts(self.portfolio.stocks, allocation)
        self.portfolio.update_stocks(allocation)
        if to_sell:
            self.trading.register(
                stamp=prices.name,
                asset=list(to_sell.keys()),
                quantity=list(to_sell.values()),
                price=prices,
                buy=False,
                fee_multiplier=self.fee_multiplier
            )
        if to_buy:
            self.trading.register(
                stamp=prices.name,
                asset=list(to_buy.keys()),
                quantity=list(to_buy.values()),
                price=prices,
                buy=True,
                fee_multiplier=self.fee_multiplier
            )
        return self._calculate_portfolio_value(prices)

    def _optimize_weights(self, prices: Series, covariance_method: str, returns_method: str, optimize: str,
                          **kwargs) -> Dict[str, float]:
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
        weights: Dict[str, float] = ef.clean_weights()
        return weights


class LstmModels(Strategy):

    def __init__(self, data: DataFrame, portfolio_value: Union[int, float], fee_multiplier: Optional[float],
                 window: int, rebalancing: Optional[int], warmup_period: int) -> None:
        super().__init__(data, portfolio_value, fee_multiplier)
        self.window = window
        self.rebalancing = rebalancing
        self.warmup_period = warmup_period

    def logic(self, counter: int, prices: Series, **kwargs) -> Optional[Union[float, np.ndarray]]:
        if counter < self.warmup_period + self.window - 1:
            return None

        if self.rebalancing is None:
            if counter == self.warmup_period + self.window - 1:
                return self._single_logic(prices, **kwargs)
            else:
                return self._calculate_portfolio_value(prices)

        if self.rebalancing is not None:
            if (counter - (self.window - 1)) % self.rebalancing == 0:
                return self._single_logic(prices, **kwargs)
            else:
                return self._calculate_portfolio_value(prices)

    def _single_logic(self, prices: Series, **kwargs) -> Union[float, np.ndarray]:
        new_weights = self._optimize_weights(prices, **kwargs)
        self.portfolio.update_weights(new_weights)
        allocation, cash = resolve_allocation(self.portfolio.weights, prices, self.portfolio_value)
        self.cash = cash
        to_sell, to_buy = resolve_order_amounts(self.portfolio.stocks, allocation)
        self.portfolio.update_stocks(allocation)
        if to_sell:
            self.trading.register(
                stamp=prices.name,
                asset=list(to_sell.keys()),
                quantity=list(to_sell.values()),
                price=prices,
                buy=False,
                fee_multiplier=self.fee_multiplier
            )
        if to_buy:
            self.trading.register(
                stamp=prices.name,
                asset=list(to_buy.keys()),
                quantity=list(to_buy.values()),
                price=prices,
                buy=True,
                fee_multiplier=self.fee_multiplier
            )
        return self._calculate_portfolio_value(prices)

    def _optimize_weights(self, prices: Series, returns_method: str, optimize: str, epochs: int, batch_size: int,
                          length: int, stopping_patience: int, **kwargs) -> Dict[str, float]:
        sliced_data = self._get_slice(current_idx=prices.name, last_observations=None)

        sample_cov = self._estimate_covariance_matrix(
            prices=sliced_data,
            epochs=epochs,
            batch_size=batch_size,
            length=length,
            stopping_patience=stopping_patience,
            **kwargs
        )

        er = expected_returns.return_model(
            prices=self._get_slice(current_idx=prices.name, last_observations=self.window), method=returns_method)
        ef = EfficientFrontier(er, sample_cov, weight_bounds=(0, 1))
        if self.fee_multiplier is not None:
            w_prev = np.fromiter(self.portfolio.weights.values(), dtype=float) if self.portfolio.weights \
                else [0] * len(self.portfolio.assets)
            ef.add_objective(transaction_cost, w_prev=w_prev, k=self.fee_multiplier)
        optimizer = getattr(ef, optimize)
        optimizer(**{k: kwargs.pop(k) for k in kwargs if k in inspect.signature(optimizer).parameters.keys()})
        weights: Dict[str, float] = ef.clean_weights()
        return weights

    def _estimate_covariance_matrix(self, prices: DataFrame, epochs: int, batch_size: int, length: int,
                                    stopping_patience: int, **kwargs) -> DataFrame:
        cov = CovarianceHandler(lookback=self.window, n_assets=len(self.assets))
        train, val = train_test_split(prices, test_size=0.2, random_state=42, shuffle=False)

        train_cholesky: DataFrame = cov.cholesky_transformation(cov.calculate_rolling_covariance_matrix(train))
        val_cholesky: DataFrame = cov.cholesky_transformation(cov.calculate_rolling_covariance_matrix(val))
        train_gen = KerasDataset(train_cholesky, length=length, batch_size=batch_size).get_generator()
        val_gen = KerasDataset(val_cholesky, length=length, batch_size=batch_size).get_generator()

        lstm = LSTM(
            shape=(np.shape(train_gen[0][0])[1], np.shape(val_gen[0][0])[2]),
            **kwargs,
        )
        lstm.compile_model(
            loss=tf.losses.mean_squared_error,
            metrics=[tf.metrics.mean_absolute_error],
            optimizer=tf.keras.optimizers.Adam()
        )
        model: Model = lstm.get_model()
        stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=stopping_patience,
            mode='auto',
        )
        model.fit(
            train_gen,
            validation_data=val_gen,
            epochs=epochs,
            batch_size=batch_size,
            verbose=1,
            shuffle=False,
            callbacks=[stopping],
        )

        predicted: DataFrame = pd.DataFrame(model.predict(val_gen), columns=val_cholesky.columns)
        predicted_matrix: DataFrame = cov.reverse_cholesky_transformation(predicted)
        predicted_matrix = predicted_matrix.tail(len(self.assets))
        predicted_matrix.index = predicted_matrix.index.droplevel(0)
        return predicted_matrix
