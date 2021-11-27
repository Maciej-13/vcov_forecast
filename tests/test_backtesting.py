import numpy as np

from numpy.testing import assert_array_equal
from vcov.modules.strategy.backtesting import Backtesting
from vcov.modules.data_handling.input_handler import InputHandler


def test_backtesting(mocker, prices):
    mocker.patch.multiple(Backtesting, __abstractmethods__=set())
    bt = Backtesting(prices, assets=['AAPL', 'F'])
    assert isinstance(bt, Backtesting)
    assert hasattr(bt, '_index')
    assert hasattr(bt, '_data')
    assert hasattr(bt, 'assets')
    assert bt.assets == ['AAPL', 'F']
    assert getattr(bt, '_index').equals(prices.index)
    assert_array_equal(getattr(bt, '_data'), prices.to_numpy())


def test_handle_data(mocker, prices):
    mocker.patch.multiple(Backtesting, __abstractmethods__=set())
    bt = Backtesting(prices, assets=['AAPL'])
    assert bt.assets == ['AAPL']
    assert_array_equal(getattr(bt, '_data').ravel(), prices["AAPL"].to_numpy())


def test_handle_data_multiple(data_dir, mocker):
    mocker.patch.multiple(Backtesting, __abstractmethods__=set())
    data = InputHandler(data_dir + '/data_short.csv', assets=['AAPL', 'BAC', 'MSFT', 'GOOG'], returns=False).get_data()
    bt = Backtesting(data, ['AAPL', 'MSFT'])
    assert bt.assets == ['AAPL', 'MSFT']
    assert getattr(bt, '_index').equals(data.index)
    prices = getattr(bt, '_data')
    assert prices.shape == (len(data.index), 2)
    assert_array_equal(prices[:, 0], data['AAPL'].values)
    assert_array_equal(prices[:, 1], data['MSFT'].values)


def test_apply_strategy(data_dir):
    data = InputHandler(data_dir + '/data_short.csv', assets=['AAPL', 'BAC', 'MSFT', 'GOOG'], returns=False).get_data()

    class MockBacktest(Backtesting):
        def logic(self, counter: int, prices: np.ndarray) -> float:
            return sum(counter * prices)

    bt = MockBacktest(data, ['AAPL', 'BAC', 'MSFT'])
    data = data[['AAPL', 'BAC', 'MSFT']]
    r = {i: sum(i * data.iloc[i, :]) for i in range(len(data))}
    results = bt.apply_strategy()
    assert len(r.keys()) == len(results)
    for i, v in enumerate(results):
        assert v == r[i]
