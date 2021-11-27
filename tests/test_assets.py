import numpy as np
import pandas as pd

from vcov.modules.data_handling.assets import Assets


def test_assets(prices):
    assets = Assets(tickers=['AAPL', 'F'], prices=prices)
    assert isinstance(assets, Assets)
    assert hasattr(assets, 'tickers')
    assert hasattr(assets, 'prices')
    assert hasattr(assets, 'returns')


def test_get_tickers(prices):
    assets = Assets(tickers=['AAPL', 'F'], prices=prices)
    assert assets.tickers == ['AAPL', 'F']


def test_get_prices(prices):
    assets = Assets(tickers=['AAPL', 'F'], prices=prices)
    assert assets.prices.equals(prices)


def test_get_returns(prices):
    assets = Assets(tickers=['AAPL', 'F'], prices=prices)
    assets_log = Assets(tickers=['AAPL', 'F'], prices=prices, log_returns=True)
    assert assets.returns.equals(prices / prices.shift(1) - 1)
    assert assets_log.returns.equals(np.log(prices / prices.shift(1)))


def test_split_by_date(prices):
    assets = Assets(tickers=['AAPL', 'F'], prices=prices)
    dt = "2021-11-04"
    old, new = assets.split_by_date(dt)
    assert len(old.prices) == 3
    assert old.prices.index[-1] == prices.index[2]
    assert len(new.prices) == 2
    assert new.prices.index[0] == pd.to_datetime(dt)


def test_split_by_length(prices):
    assets = Assets(tickers=['AAPL', 'F'], prices=prices)
    old, new = assets.split_by_length(split=0.8)
    assert len(old.prices) == 4
    assert len(new.prices) == 1
