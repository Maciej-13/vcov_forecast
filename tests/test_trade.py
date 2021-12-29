import io
import os
import pickle

from unittest.mock import patch

import numpy as np
import pandas as pd

from vcov.modules.trade.trade import Trade, TradeHistory


def test_trade():
    t = Trade('test', 10, 2.5, True, 0.01)
    assert t.asset == 'test'
    assert t.quantity == 10
    assert t.price == 2.5
    assert t.buy is True
    assert t.fee_multiplier == 0.01


@patch('sys.stdout', new_callable=io.StringIO)
def test_trade_print(mock_stdout):
    print(Trade('test', 10, 2.5, True, 0.01))
    assert mock_stdout.getvalue() == "asset: test\nquantity: 10\nprice: 2.5\nposition: buy\nfee: 0.01\n"


def test_trade_calculate_fees():
    t = Trade('test', 10, 2.5, True, 0.01)
    assert t.calculate_fees() == 10 * 2.5 * 0.01


def test_trade_history():
    history = TradeHistory()
    assert isinstance(history, TradeHistory)
    assert not history.history
    assert isinstance(history.history, dict)


def test_trade_history_register():
    history = TradeHistory()
    dts = pd.date_range('01-01-2021', '01-02-2021')
    history.register(dts[0], 'test', 10, 2.5, True, 0.01)
    history.register(dts[0], 'test2', 5, 0.5, False, 0.02)
    history.register(dts[1], 'test', 10, 2.5, False, 0.01)
    history.register(dts[1], 'test2', 5, 0.5, True, 0.02)

    assert history.history[dts[0]][0].asset == 'test'
    assert history.history[dts[0]][0].quantity == 10
    assert history.history[dts[0]][0].price == 2.5
    assert history.history[dts[0]][0].buy is True
    assert history.history[dts[0]][0].fee_multiplier == 0.01

    assert history.history[dts[0]][1].asset == 'test2'
    assert history.history[dts[0]][1].quantity == 5
    assert history.history[dts[0]][1].price == .5
    assert history.history[dts[0]][1].buy is False
    assert history.history[dts[0]][1].fee_multiplier == 0.02

    assert history.history[dts[1]][0].asset == 'test'
    assert history.history[dts[1]][0].quantity == 10
    assert history.history[dts[1]][0].price == 2.5
    assert history.history[dts[1]][0].buy is False
    assert history.history[dts[1]][0].fee_multiplier == 0.01

    assert history.history[dts[1]][1].asset == 'test2'
    assert history.history[dts[1]][1].quantity == 5
    assert history.history[dts[1]][1].price == .5
    assert history.history[dts[1]][1].buy is True
    assert history.history[dts[1]][1].fee_multiplier == 0.02


def test_trade_history_register_multiple():
    history = TradeHistory()
    dt = pd.to_datetime('06-23-2020')
    history.register(dt, ['test', 'test2'], [10, 5], np.array([2.5, 5]), True, 0.1)

    assert history.history[dt][0].asset == 'test'
    assert history.history[dt][0].quantity == 10
    assert history.history[dt][0].price == 2.5
    assert history.history[dt][0].buy is True
    assert history.history[dt][0].fee_multiplier == 0.1

    assert history.history[dt][1].asset == 'test2'
    assert history.history[dt][1].quantity == 5
    assert history.history[dt][1].price == 5
    assert history.history[dt][1].buy is True
    assert history.history[dt][1].fee_multiplier == 0.1

    dt = pd.to_datetime('06-24-2020')
    history.register(dt, 'test3', 1, 100, True, 0.01)
    history.register(dt, ['test', 'test2'], [10, 5], np.array([2.5, 5]), False, 0.1)

    assert history.history[dt][0].asset == 'test3'
    assert history.history[dt][0].quantity == 1
    assert history.history[dt][0].price == 100
    assert history.history[dt][0].buy is True
    assert history.history[dt][0].fee_multiplier == 0.01

    assert history.history[dt][1].asset == 'test'
    assert history.history[dt][1].quantity == 10
    assert history.history[dt][1].price == 2.5
    assert history.history[dt][1].buy is False
    assert history.history[dt][1].fee_multiplier == 0.1

    assert history.history[dt][2].asset == 'test2'
    assert history.history[dt][2].quantity == 5
    assert history.history[dt][2].price == 5
    assert history.history[dt][2].buy is False
    assert history.history[dt][2].fee_multiplier == 0.1


def test_trade_history_save(data_dir):
    dts = pd.date_range('01-01-2021', '01-02-2021')
    history = TradeHistory()
    history.register(dts[0], 'test', 10, 2.5, True, 0.01)
    history.register(dts[0], 'test2', 5, 0.5, False, 0.02)
    history.register(dts[1], 'test', 10, 2.5, False, 0.01)
    history.register(dts[1], 'test2', 5, 0.5, True, 0.02)
    history.save(f'{data_dir}/test')
    assert os.path.exists(f'{data_dir}/test.pickle')

    with open(f'{data_dir}/test.pickle', 'rb') as f:
        h = pickle.load(f)

    assert h[dts[0]][0] == history.history[dts[0]][0]
    assert h[dts[0]][1] == history.history[dts[0]][1]
    assert h[dts[1]][0] == history.history[dts[1]][0]
    assert h[dts[1]][1] == history.history[dts[1]][1]

    os.remove(f'{data_dir}/test.pickle')
