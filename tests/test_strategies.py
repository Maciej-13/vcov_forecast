import numpy as np
import pandas as pd

from vcov.modules.strategy.strategies import EquallyWeighted, resolve_allocation, RiskModels, resolve_order_amounts
from vcov.modules.portfolio.portfolio import Portfolio


def test_resolve_allocation():
    allocation, cash = resolve_allocation({'A': 0.5, 'B': 0.25, 'C': 0.25},
                                          pd.Series([90.0, 20.0, 50.0], index=['A', 'B', 'C']), 1000.0)
    allocation_2, cash_2 = resolve_allocation({'A': 0.5, 'B': 0.25, 'C': 0.25},
                                              pd.Series([90.0, 20.0, 1000.0], index=['A', 'B', 'C']), 1000.0)
    assert allocation == {'A': 5, 'B': 12, 'C': 5}
    assert allocation_2 == {'A': 5, 'B': 12}
    assert cash == 1000 - allocation['A'] * 90 - allocation['B'] * 20 - allocation['C'] * 50
    assert cash_2 == 1000 - allocation['A'] * 90 - allocation['B'] * 20


def test_resolve_order_amounts():
    old_stocks = {'a': 5, 'b': 2, 'c': 3, 'd': 10}
    new_stocks = {'a': 1, 'b': 7, 'c': 35, 'h': 5}
    sell, buy = resolve_order_amounts(old_stocks, new_stocks)
    assert sell == {'a': 4, 'd': 10}
    assert buy == {'b': 5, 'c': 32, 'h': 5}


def test_equally_weighted(multiple_prices, asset_names):
    strategy = EquallyWeighted(multiple_prices, 1000.0, None)
    assert hasattr(strategy, "portfolio")
    assert isinstance(strategy.portfolio, Portfolio)
    assert strategy.portfolio.assets == asset_names
    assert strategy.assets == asset_names
    assert hasattr(strategy, '_data')
    data = getattr(strategy, '_data')
    pd.testing.assert_frame_equal(data, multiple_prices)
    assert hasattr(strategy, 'portfolio_value')
    assert isinstance(strategy.portfolio_value, float)
    assert hasattr(strategy, 'cash')
    assert isinstance(strategy.cash, float)
    assert hasattr(strategy, 'trading')


def test_equally_weighted_portfolio_characteristics(multiple_prices, asset_names):
    strategy = EquallyWeighted(multiple_prices, 1000, None)
    _ = strategy.logic(0, multiple_prices.iloc[0, :])
    assert strategy.portfolio.weights == {i: 0.25 for i in asset_names}
    allocation = resolve_allocation(weights={i: 0.25 for i in asset_names},
                                    portfolio_value=1000, prices=multiple_prices.iloc[0])
    assert strategy.portfolio.stocks == allocation[0]
    assert strategy.cash == allocation[1]


def test_equally_weighted_logic(multiple_prices, asset_names):
    strategy = EquallyWeighted(multiple_prices, 1000, None)
    portfolio_value = strategy.logic(0, multiple_prices.iloc[0, :])
    allocation, cash = resolve_allocation(weights={i: 0.25 for i in asset_names},
                                          portfolio_value=1000, prices=multiple_prices.iloc[0])
    assert portfolio_value == cash + np.dot(np.fromiter(allocation.values(), dtype=float),
                                            multiple_prices.iloc[0][allocation.keys()])


def test_equally_weighted_portfolio_apply_strategy(multiple_prices, asset_names):
    strategy = EquallyWeighted(multiple_prices, 1000, None)
    results = strategy.apply_strategy()
    allocation, cash = resolve_allocation(weights={i: 0.25 for i in asset_names},
                                          portfolio_value=1000, prices=multiple_prices.iloc[0])
    multiple_prices = multiple_prices[allocation.keys()]
    weighted_prices = multiple_prices['AAPL'] * allocation['AAPL'] + multiple_prices['BAC'] * allocation['BAC'] + \
        multiple_prices['MSFT'] * allocation['MSFT'] + cash
    assert all(round(i, 10) == round(j, 10) for i, j in zip(results, weighted_prices))


def test_equally_weighted_portfolio_trades(multiple_prices):
    strategy = EquallyWeighted(multiple_prices, 1000, None)
    _ = strategy.apply_strategy()

    assert isinstance(strategy.trading.history, dict)
    assert list(strategy.trading.history.keys())[0] == multiple_prices.index[0]

    dt = multiple_prices.index[0]

    assert strategy.trading.history[dt][0].asset == 'AAPL'
    assert strategy.trading.history[dt][1].asset == 'BAC'
    assert strategy.trading.history[dt][2].asset == 'MSFT'

    assert strategy.trading.history[dt][0].price == multiple_prices.iloc[0, 0]
    assert strategy.trading.history[dt][1].price == multiple_prices.iloc[0, 1]
    assert strategy.trading.history[dt][2].price == multiple_prices.iloc[0, 2]

    assert strategy.trading.history[dt][0].quantity == 9
    assert strategy.trading.history[dt][1].quantity == 14
    assert strategy.trading.history[dt][2].quantity == 4

    assert all(strategy.trading.history[dt][i].buy is True for i in range(3))
    assert all(strategy.trading.history[dt][i].fee_multiplier is None for i in range(3))


def test_risk_models_sample_covariance(multiple_prices):
    strategy = RiskModels(multiple_prices, 1000, window=30, rebalancing=20, fee_multiplier=None)
    results = strategy.apply_strategy(
        covariance_method='sample_cov',
        returns_method='mean_historical_return',
        optimize='min_volatility',
        frequency=252,
    )
    assert len(results) == len(multiple_prices)
    assert len(results.dropna()) == len(multiple_prices) - 29


def test_risk_models_optimize_weights(multiple_prices):
    strategy = RiskModels(multiple_prices, 1000, window=30, rebalancing=None, fee_multiplier=None)
    assert not list(strategy.portfolio.weights.values())
    weights = strategy._optimize_weights(multiple_prices.iloc[29, :], 'sample_cov', 'mean_historical_return',
                                         'min_volatility')
    assert round(sum(weights.values()), 4) == 1


def test_risk_models_single_logic(multiple_prices):
    strategy = RiskModels(multiple_prices, 1000, window=30, rebalancing=None, fee_multiplier=None)
    assert not strategy.trading.history
    strategy._single_logic(
        multiple_prices.iloc[29, :],
        covariance_method='sample_cov',
        returns_method='mean_historical_return',
        optimize='min_volatility'
    )
    history = strategy.trading.history
    dt = multiple_prices.index[29]
    assert history
    assert history[dt]
    assert history[dt][0].asset == 'AAPL'
    assert history[dt][1].asset == 'BAC'
    assert sum(history[dt][i].price * history[dt][i].quantity for i in range(len(history[dt]))) < 1000
