import numpy as np
import pandas as pd

from vcov.modules.data_handling.assets import Assets


def test_assets(prices):
    assets = Assets(names=['AAPL', 'F'], prices=prices)
    assert isinstance(assets, Assets)
    assert hasattr(assets, '_names')
    assert hasattr(assets, '_prices')
    assert hasattr(assets, '_returns')


def test_get_names(prices):
    assets = Assets(names=['AAPL', 'F'], prices=prices)
    assert assets.get_names() == ['AAPL', 'F']


def test_get_prices(prices):
    assets = Assets(names=['AAPL', 'F'], prices=prices)
    assert assets.get_prices().equals(prices)


def test_get_returns(prices):
    assets = Assets(names=['AAPL', 'F'], prices=prices)
    assets_log = Assets(names=['AAPL', 'F'], prices=prices, log_returns=True)
    assert assets.get_returns().equals(prices / prices.shift(1) - 1)
    assert assets_log.get_returns().equals(np.log(prices / prices.shift(1)))


def test_split_by_date(prices):
    assets = Assets(names=['AAPL', 'F'], prices=prices)
    dt = "2021-11-04"
    old, new = assets.split_by_date(dt)
    assert len(old.get_prices()) == 3
    assert old.get_prices().index[-1] == prices.index[2]
    assert len(new.get_prices()) == 2
    assert new.get_prices().index[0] == pd.to_datetime(dt)


def test_split_by_length(prices):
    assets = Assets(names=['AAPL', 'F'], prices=prices)
    old, new = assets.split_by_length(split=0.8)
    assert len(old.get_prices()) == 4
    assert len(new.get_prices()) == 1
