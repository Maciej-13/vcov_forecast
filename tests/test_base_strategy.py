import pandas as pd

from vcov.modules.strategy.base_strategy import Strategy


def test_backtesting(mocker, prices):
    mocker.patch.multiple(Strategy, __abstractmethods__=set())
    bt = Strategy(prices, portfolio_value=100, fee_multiplier=0.1)
    assert isinstance(bt, Strategy)
    assert hasattr(bt, '_data')
    assert hasattr(bt, 'assets')
    assert bt.assets == ['AAPL', 'F']
    assert getattr(bt, '_data').index.equals(prices.index)
    pd.testing.assert_frame_equal(getattr(bt, '_data'), prices)
    assert bt.portfolio_value == 100
    assert bt.fee_multiplier == 0.1


def test_get_slice(multiple_prices, mocker):
    mocker.patch.multiple(Strategy, __abstractmethods__=set())
    bt = Strategy(multiple_prices, portfolio_value=100, fee_multiplier=0.1)
    for i, idx in enumerate(multiple_prices.iloc[100:].index):
        pd.testing.assert_frame_equal(bt._get_slice(idx, 100), multiple_prices.iloc[i + 1: i + 101, :])


def test_apply_strategy(multiple_prices):
    class MockBacktest(Strategy):
        def logic(self, counter: int, prices: pd.Series, **kwargs) -> float:
            return sum(counter * prices)

    data = multiple_prices[['AAPL', 'BAC', 'MSFT']]
    bt = MockBacktest(data, portfolio_value=100, fee_multiplier=0.1)
    r = {i: sum(i * data.iloc[i, :]) for i in range(len(data))}
    results = bt.apply_strategy()
    assert len(r.keys()) == len(results)
    for i, v in enumerate(results):
        assert v == r[i]
