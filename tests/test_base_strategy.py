import numpy as np
import pandas
import pandas as pd

from numpy.testing import assert_array_equal
from vcov.modules.strategy.base_strategy import Strategy


def test_backtesting(mocker, prices):
    mocker.patch.multiple(Strategy, __abstractmethods__=set())
    bt = Strategy(prices, portfolio_value=100, fee_multiplier=0.1)
    assert isinstance(bt, Strategy)
    assert hasattr(bt, '_index')
    assert hasattr(bt, '_data')
    assert hasattr(bt, 'assets')
    assert bt.assets == ['AAPL', 'F']
    assert getattr(bt, '_index').equals(prices.index)
    assert_array_equal(getattr(bt, '_data'), prices.to_numpy())
    assert bt.portfolio_value == 100
    assert bt.fee_multiplier == 0.1


def test_handle_data(mocker, prices):
    mocker.patch.multiple(Strategy, __abstractmethods__=set())
    bt = Strategy(prices, portfolio_value=100, fee_multiplier=0.1)
    assert bt.assets == ['AAPL', 'F']
    assert_array_equal(getattr(bt, '_data'), prices.to_numpy())
    pd.testing.assert_index_equal(getattr(bt, '_index'), prices.index)


def test_handle_data_multiple(multiple_prices, mocker):
    multiple_prices = multiple_prices[['AAPL', 'MSFT']]
    mocker.patch.multiple(Strategy, __abstractmethods__=set())
    bt = Strategy(multiple_prices, portfolio_value=10, fee_multiplier=0.1)
    assert bt.assets == ['AAPL', 'MSFT']
    assert getattr(bt, '_index').equals(multiple_prices.index)
    prices = getattr(bt, '_data')
    assert prices.shape == (len(multiple_prices.index), 2)
    assert_array_equal(prices[:, 0], multiple_prices['AAPL'].values)
    assert_array_equal(prices[:, 1], multiple_prices['MSFT'].values)


def test_get_slice(multiple_prices, mocker):
    mocker.patch.multiple(Strategy, __abstractmethods__=set())
    bt = Strategy(multiple_prices, portfolio_value=100, fee_multiplier=0.1)
    for i in range(100, len(multiple_prices)):
        np.testing.assert_array_equal(bt._get_slice(i, 100), multiple_prices[i - 99: i + 1].values)


def test_get_slice_expanding(multiple_prices, mocker):
    mocker.patch.multiple(Strategy, __abstractmethods__=set())
    bt = Strategy(multiple_prices, portfolio_value=100, fee_multiplier=0.1)
    for i in range(1, len(multiple_prices)):
        s = i - 99 if i >= 99 else 0
        np.testing.assert_array_equal(bt._get_slice(i, 100), multiple_prices[s: i + 1].values)


def test_apply_strategy(multiple_prices):
    class MockBacktest(Strategy):
        def logic(self, counter: int, prices: np.ndarray) -> float:
            return sum(counter * prices)

    data = multiple_prices[['AAPL', 'BAC', 'MSFT']]
    bt = MockBacktest(data, portfolio_value=100, fee_multiplier=0.1)
    r = {i: sum(i * data.iloc[i, :]) for i in range(len(data))}
    results = bt.apply_strategy()
    assert len(r.keys()) == len(results)
    for i, v in enumerate(results):
        assert v == r[i]
