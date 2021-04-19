import unittest
import pandas as pd

from vcov_forecast.modules.data_handling.data_reading import YahooDataReader
from vcov_forecast.modules.data_handling.data_reading import YahooReader


class TestYahooDataReader(unittest.TestCase):

    def setUp(self):
        self.yreader = YahooDataReader("AAPL", start="2021-01-04", end="2021-01-14")

    def test_get_data(self):
        data = self.yreader.get_data()
        self.assertIsInstance(data, pd.DataFrame)
        self.assertEqual(len(data.index), 8)
        self.assertEqual(data.index[0], pd.Timestamp('2021-01-04'))

    def test_get_ticker(self):
        self.assertEqual(self.yreader.get_ticker(), "AAPL")

    def test_get_high(self):
        data = self.yreader.get_high()
        self.assertEqual(round(data[0], 3), 133.61)
        self.assertEqual(data.name, "High")

    def test_get_low(self):
        data = self.yreader.get_low()
        self.assertEqual(round(data[0], 3), 126.76)
        self.assertEqual(data.name, "Low")

    def test_get_close(self):
        data = self.yreader.get_close()
        self.assertEqual(round(data[0], 3), 129.41)
        self.assertEqual(data.name, "Close")

    def test_get_open(self):
        data = self.yreader.get_open()
        self.assertEqual(round(data[0], 3), 133.52)
        self.assertEqual(data.name, "Open")

    def test_get_adj_close(self):
        data = self.yreader.get_adj_close()
        self.assertEqual(round(data[0], 3), 129.217)
        self.assertEqual(data.name, "Adj Close")

    def test_get_volume(self):
        data = self.yreader.get_volume()
        self.assertEqual(data[0], 143301900)
        self.assertEqual(data.name, "Volume")


class TestYahooReader(unittest.TestCase):

    def setUp(self) -> None:
        self.yreader = YahooReader('AAPL NFLX', start="2021-01-04", end="2021-01-14")

    def test_get_data(self):
        data = self.yreader.get_data()
        self.assertIsInstance(data, pd.DataFrame)
        self.assertEqual(len(data), 8)

    def test_get_all_tickers(self):
        tickers = self.yreader.get_all_tickers()
        self.assertEqual(len(tickers), 2)
        for tick in tickers:
            self.assertIn(tick, ['AAPL', 'NFLX'])

    def test_get_columns(self):
        data = self.yreader.get_columns('Open')
        self.assertEqual(len(data.columns), 2)
        self.assertEqual(list(data.columns[0])[0], 'Open')
        data = self.yreader.get_columns('Open', single_index=True)
        self.assertEqual(data.columns[0].split(' ')[0], 'Open')

    def test_get_data_by_tickers(self):
        data = self.yreader.get_data_by_tickers('NFLX', single_index=True)
        self.assertEqual(len(data.columns), 6)
        self.assertEqual(data.columns[0].split(' ')[-1], 'NFLX')

    '''def test_save(self):
        self.yreader.save(path='../data', single_index=True, single_file=False)'''
    # I checked properties of save method manually and commented these lines to protect any data from being accidentally
    # overwritten due to execution of this method
