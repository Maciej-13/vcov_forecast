import os
import pandas as pd

from vcov.modules.data_handling.data_reading import DataReader


def test_reader():
    reader = DataReader('AAPL NFLX', start="2021-11-01", end="2021-11-15")
    assert reader.tickers == ['AAPL', 'NFLX']


def test_get_data():
    reader = DataReader('AAPL NFLX', start="2021-11-01", end="2021-11-15")
    data = reader.get_data()
    assert isinstance(data, pd.DataFrame)
    assert len(data) == 10


def test_get_columns():
    reader = DataReader('AAPL NFLX', start="2021-11-01", end="2021-11-15")
    data = reader.get_columns('Open')
    assert len(data.columns) == 2
    assert list(data.columns[0])[0] == 'Open'
    data = reader.get_columns('Open', single_index=True)
    assert data.columns[0].split(' ')[0] == 'Open'


def test_get_data_by_tickers():
    reader = DataReader('AAPL NFLX', start="2021-11-01", end="2021-11-15")
    data = reader.get_data_by_tickers('NFLX', single_index=True)
    assert len(data.columns) == 6
    assert data.columns[0].split(' ')[-1] == 'NFLX'


def test_save(data_dir):
    reader = DataReader('AAPL NFLX', start="2021-11-01", end="2021-11-15")
    reader.save(path=data_dir, single_index=True, single_file=True)
    assert 'AAPL_NFLX.csv' in os.listdir(data_dir)
    data = pd.read_csv(data_dir + '/AAPL_NFLX.csv', index_col=0)
    assert (reader._data.values == data.values).all()
    os.remove(data_dir + "/AAPL_NFLX.csv")


def test_validate_tickers():
    tkr = 'A B C'
    tkrs = ['A', 'B', 'C']
    assert DataReader._validate_tickers(tkr) == DataReader._validate_tickers(tkrs) == ['A', 'B', 'C']


def test_modify_tickers_type():
    tkr = 'A'
    tkrs = ['A', 'V']
    reader = DataReader('AAPL NFLX', start="2021-11-01", end="2021-11-15")
    assert reader._modify_tickers_type(tkr) == ['A']
    assert reader._modify_tickers_type(tkrs) == ['A', 'V']
