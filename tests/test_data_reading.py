import unittest
import pandas as pd

from vcov_forecast.modules.data_handling.data_reading import YahooDataReader


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
        self.assertEqual(round(data[0], 3), 129.41)
        self.assertEqual(data.name, "Adj Close")

    def test_get_volume(self):
        data = self.yreader.get_volume()
        self.assertEqual(data[0], 143301900)
        self.assertEqual(data.name, "Volume")

