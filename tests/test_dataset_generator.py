import unittest
import pandas as pd
import numpy as np

from vcov_forecast.modules.data_handling.dataset_generator import *


class TestInputHandler(unittest.TestCase):

    def setUp(self) -> None:
        self.tickers = ['AAPL', 'BAC', 'MSFT', 'GOOG']
        self.handler = InputHandler('../data/data_short.csv', self.tickers, 'Close', returns=False)
        temp_data = pd.read_csv("../data/data_short.csv", parse_dates=['Date'], index_col='Date')
        to_select = ['Close ' + t for t in self.tickers]
        temp_data = temp_data[to_select]
        temp_data.columns = self.tickers
        self.temp_data = temp_data

    def test_get_data(self):
        self.assertIsInstance(self.handler.get_data(), pd.DataFrame)
        self.assertEqual(len(self.handler.get_data().columns), 4)
        self.assertCountEqual(self.handler.get_data().columns.tolist(), self.tickers)
        self.assertIsInstance(self.handler.get_data().index, pd.DatetimeIndex)

    def test_select_column(self):
        pd.testing.assert_frame_equal(self.temp_data, self.handler.get_data())

    def test_calculate_returns(self):
        handler_data = InputHandler('../data/data_short.csv', self.tickers, returns=True).get_data()
        for col in handler_data.columns:
            temp_returns = np.diff(self.temp_data[col]) / self.temp_data.loc[:self.temp_data.index[-2], col]
            np.testing.assert_array_almost_equal(temp_returns.values, handler_data[col].values, decimal=16)


class TestCovarianceHandler(unittest.TestCase):

    def setUp(self) -> None:
        self.cov = CovarianceHandler(lookback=15, n_assets=4)
        self.data = InputHandler('../data/data_short.csv', assets=['AAPL', 'BAC', 'MSFT', 'GOOG'], column='Close',
                                 returns=True).get_data()
        self.data_idx = self.data.reset_index(drop=True)

    def test_calculate_rolling_covariance_matrix(self):
        covariances = self.cov.calculate_rolling_covariance_matrix(self.data)
        for i in self.data_idx.index[14:]:
            temp_data = self.data_idx.loc[i - 14:i]
            np.testing.assert_array_almost_equal(np.cov(temp_data.to_numpy().T), covariances.loc[
                covariances.index.get_level_values(0).unique()[i - 14]].to_numpy(), decimal=10)

    def test_split_covariance_matrices(self):
        dates = self.data.index[14:]
        cov_by_dt = self.cov.split_covariance_matrices(self.cov.calculate_rolling_covariance_matrix(self.data))
        for i in range(14, len(self.data)):
            temp_data = self.data_idx.loc[i - 14:i]
            np.testing.assert_array_almost_equal(np.cov(temp_data.to_numpy().T)[np.tril_indices(4)],
                                                 cov_by_dt[dates[i-14]], decimal=10)

    def test_split_covariance_to_long(self):
        long_cov = self.cov.split_covariance_to_long(self.cov.calculate_rolling_covariance_matrix(self.data))
        for i in range(14, len(self.data)):
            cov = np.cov(self.data_idx.loc[i - 14:i].to_numpy().T)[np.tril_indices(4)].ravel()
            np.testing.assert_array_almost_equal(cov, long_cov.loc[self.data.index[i]].to_numpy())

    def test_get_names(self):
        tickers = ['AMZN', 'AAPL', 'MSFT', 'GOOG', 'AEE', 'ANSS',
                   'CDNS', 'CSCO', 'CTSH', 'DXC', 'FISV', 'FLT']
        nms = self.cov.get_names(tickers)
        bench = []
        for nm in tickers:
            for nm_two in tickers:
                if '_'.join((nm, nm_two)) not in bench:
                    bench.append('_'.join((nm_two, nm)))

        self.assertCountEqual(bench, nms)
