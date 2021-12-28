import numpy as np

from vcov.modules.strategy.strategies import EquallyWeighted, resolve_allocation
from vcov.modules.portfolio.portfolio import Portfolio


def test_resolve_allocation():
    allocation = resolve_allocation(['A', 'B', 'C'], {'A': 0.5, 'B': 0.25, 'C': 0.25},
                                    np.array([90.0, 20.0, 50.0]), 1000.0)
    allocation_2 = resolve_allocation(['A', 'B', 'C'], {'A': 0.5, 'B': 0.25, 'C': 0.25},
                                      np.array([90.0, 20.0, 1000.0]), 1000.0)
    assert allocation == [5, 12, 5]
    assert allocation_2 == [5, 12, 0]


def test_equally_weighted(multiple_prices, asset_names):
    strategy = EquallyWeighted(multiple_prices, 1000, None)
    assert hasattr(strategy, "portfolio")
    assert isinstance(strategy.portfolio, Portfolio)
    assert strategy.portfolio.assets == asset_names
    assert strategy.assets == asset_names
    assert hasattr(strategy, '_data')
    data = getattr(strategy, '_data')
    np.testing.assert_array_equal(data, multiple_prices.values)


def test_equally_weighted_logic(multiple_prices):
    strategy = EquallyWeighted(multiple_prices, 1000, None)
    weighted_price = strategy.logic(0, multiple_prices.iloc[0, :].values)
    wp = sum(multiple_prices.iloc[0, :].values * 0.25)
    assert weighted_price == wp


def test_equally_weighted_portfolio_characteristics(multiple_prices, asset_names):
    strategy = EquallyWeighted(multiple_prices, 1000, None)
    _ = strategy.logic(0, multiple_prices.iloc[0, :].values)
    assert strategy.portfolio.weights == {i: 0.25 for i in asset_names}


def test_equally_weighted_portfolio_apply_strategy(multiple_prices):
    strategy = EquallyWeighted(multiple_prices, 1000, None)
    results = strategy.apply_strategy()
    weighted_prices = (multiple_prices * 1 / 4).sum(axis=1).to_list()
    assert all(round(i, 10) == round(j, 10) for i, j in zip(results, weighted_prices))


def test_equally_weighted_portfolio_trades(multiple_prices):
    strategy = EquallyWeighted(multiple_prices, 1000, None)
    _ = strategy.apply_strategy()
    assert isinstance(strategy.trading.history, dict)
    assert list(strategy.trading.history.keys()) == [0]

    assert strategy.trading.history[0][0].asset == 'AAPL'
    assert strategy.trading.history[0][1].asset == 'BAC'
    assert strategy.trading.history[0][2].asset == 'MSFT'
    assert strategy.trading.history[0][3].asset == 'GOOG'

    assert strategy.trading.history[0][0].price == multiple_prices.iloc[0, 0]
    assert strategy.trading.history[0][1].price == multiple_prices.iloc[0, 1]
    assert strategy.trading.history[0][2].price == multiple_prices.iloc[0, 2]
    assert strategy.trading.history[0][3].price == multiple_prices.iloc[0, 3]

    assert strategy.trading.history[0][0].quantity == 9
    assert strategy.trading.history[0][1].quantity == 14
    assert strategy.trading.history[0][2].quantity == 4
    assert strategy.trading.history[0][3].quantity == 0

    assert all(strategy.trading.history[0][i].buy is True for i in range(4))
    assert all(strategy.trading.history[0][i].fee_multiplier is None for i in range(4))
