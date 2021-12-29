import pandas as pd

from vcov.modules.strategy.strategies import EquallyWeighted, resolve_allocation, RiskModels
from vcov.modules.portfolio.portfolio import Portfolio


def test_resolve_allocation():
    allocation = resolve_allocation({'A': 0.5, 'B': 0.25, 'C': 0.25},
                                    pd.Series([90.0, 20.0, 50.0], index=['A', 'B', 'C']), 1000.0)
    allocation_2 = resolve_allocation({'A': 0.5, 'B': 0.25, 'C': 0.25},
                                      pd.Series([90.0, 20.0, 1000.0], index=['A', 'B', 'C']), 1000.0)
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
    pd.testing.assert_frame_equal(data, multiple_prices)


def test_equally_weighted_portfolio_characteristics(multiple_prices, asset_names):
    strategy = EquallyWeighted(multiple_prices, 1000, None)
    _ = strategy.logic(0, multiple_prices.iloc[0, :])
    assert strategy.portfolio.weights == {i: 0.25 for i in asset_names}


def test_equally_weighted_logic(multiple_prices):
    strategy = EquallyWeighted(multiple_prices, 1000, None)
    weighted_price = strategy.logic(0, multiple_prices.iloc[0, :])
    wp = sum(multiple_prices.iloc[0, :].values * 0.25)
    assert weighted_price == wp


def test_equally_weighted_portfolio_apply_strategy(multiple_prices):
    strategy = EquallyWeighted(multiple_prices, 1000, None)
    results = strategy.apply_strategy()
    weighted_prices = (multiple_prices * 1 / 4).sum(axis=1).to_list()
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
    strategy._optimize_weights(multiple_prices.iloc[29, :], 'sample_cov', 'mean_historical_return', 'min_volatility')
    assert round(sum(strategy.portfolio.weights.values()), 4) == 1


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
