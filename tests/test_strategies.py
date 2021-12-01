import numpy as np

from vcov.modules.strategy.strategies import EquallyWeighted
from vcov.modules.strategy.portfolio import Portfolio


def test_equally_weighted(multiple_prices, asset_names):
    strategy = EquallyWeighted(multiple_prices, asset_names)
    assert hasattr(strategy, "portfolio")
    assert isinstance(strategy.portfolio, Portfolio)
    assert strategy.portfolio.assets == asset_names
    assert strategy.assets == asset_names
    assert hasattr(strategy, '_data')
    data = getattr(strategy, '_data')
    np.testing.assert_array_equal(data, multiple_prices.values)


def test_equally_weighted_logic(multiple_prices, asset_names):
    strategy = EquallyWeighted(multiple_prices, asset_names)
    weighted_price = strategy.logic(0, multiple_prices.iloc[0, :].values)
    wp = sum(multiple_prices.iloc[0, :].values * 0.25)
    assert weighted_price == wp


def test_equally_weighted_portfolio_characteristics(multiple_prices, asset_names):
    strategy = EquallyWeighted(multiple_prices, asset_names)
    _ = strategy.logic(0, multiple_prices.iloc[0, :].values)
    assert strategy.portfolio.weights == {i: 0.25 for i in asset_names}
    assert strategy.portfolio.buy_prices == {i: multiple_prices.loc[multiple_prices.index[0], i] for i in asset_names}


def test_equally_weighted_portfolio_apply_strategy(multiple_prices, asset_names):
    strategy = EquallyWeighted(multiple_prices, asset_names)
    results = strategy.apply_strategy()
    weighted_prices = (multiple_prices * 1 / 4).sum(axis=1).to_list()
    assert all(round(i, 10) == round(j, 10) for i, j in zip(results, weighted_prices))
