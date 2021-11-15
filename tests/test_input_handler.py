import pandas as pd

from vcov.modules.data_handling.input_handler import InputHandler


def test_input_handler(data_path):
    handler = InputHandler(path=data_path, assets=['AAPL', 'F'], column='Close', returns=True)
    assert handler.assets == ['AAPL', 'F']


def test_input_handler_get_data(data_path):
    data = InputHandler(path=data_path, assets=['AAPL', 'F'], column='Close', returns=False).get_data()
    assert isinstance(data, pd.DataFrame)
    assert len(data.columns) == 2
    assert data.columns.to_list() == ['AAPL', "F"]
    assert len(data) == 5


def test_input_handler_get_data_returns(data_path):
    data_r = InputHandler(path=data_path, assets=['AAPL', 'F'], column='Close', returns=True).get_data()
    data = InputHandler(path=data_path, assets=['AAPL', 'F'], column='Close', returns=False).get_data()
    assert isinstance(data_r, pd.DataFrame)
    assert len(data_r.columns) == 2
    assert data_r.columns.to_list() == ['AAPL', "F"]
    assert len(data_r) == 4
    assert data_r.equals(data.pct_change(1).dropna())


def test_input_handler_split(data_path):
    train, test = InputHandler(path=data_path, assets=['AAPL', 'F'],
                               column='Close', returns=True).train_test_split(0.2)
    assert isinstance(train, pd.DataFrame)
    assert isinstance(test, pd.DataFrame)
    assert len(test) + len(train) == 4
