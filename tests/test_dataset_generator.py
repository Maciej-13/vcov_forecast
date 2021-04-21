import unittest

import numpy as np
import pandas as pd

from scipy import linalg

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
        self.rolling_cov = self.cov.calculate_rolling_covariance_matrix(self.data)

    def test_calculate_rolling_covariance_matrix(self):
        covariances = self.cov.calculate_rolling_covariance_matrix(self.data)
        for i in self.data_idx.index[14:]:
            temp_data = self.data_idx.loc[i - 14:i]
            np.testing.assert_array_almost_equal(np.cov(temp_data.to_numpy().T), covariances.loc[
                covariances.index.get_level_values(0).unique()[i - 14]].to_numpy(), decimal=16)

    def test_split_covariance_matrices(self):
        dates = self.data.index[14:]
        cov_by_dt = self.cov.split_covariance_matrices(self.rolling_cov)
        for i in range(14, len(self.data)):
            temp_data = self.data_idx.loc[i - 14:i]
            np.testing.assert_array_almost_equal(np.cov(temp_data.to_numpy().T)[np.tril_indices(4)],
                                                 cov_by_dt[dates[i - 14]], decimal=16)

    def test_split_covariance_to_wide(self):
        long_cov = self.cov.split_covariance_to_wide(self.rolling_cov)
        for i in range(14, len(self.data)):
            cov = np.cov(self.data_idx.loc[i - 14:i].to_numpy().T)[np.tril_indices(4)].ravel()
            np.testing.assert_array_almost_equal(cov, long_cov.loc[self.data.index[i]].to_numpy())

    def test_get_covariance_vector(self):
        vector = self.cov.get_covariance_vector(self.rolling_cov, 'BAC_AAPL')
        filtered_data = self.data_idx.loc[:, ['AAPL', 'BAC']]
        for i in range(14, len(self.data)):
            cov = np.cov(filtered_data.loc[i - 14:i].to_numpy().T)[0][1].ravel()
            np.testing.assert_array_almost_equal(cov, vector.loc[self.data.index[i]])

    def test_cholesky_transformation(self):
        cholesky = self.cov.cholesky_transformation(self.rolling_cov, return_dict=True)
        cholesky_df = self.cov.cholesky_transformation(self.rolling_cov)
        idx = np.tril_indices(4)
        self.assertIsInstance(cholesky, dict)
        self.assertIsInstance(cholesky_df, pd.DataFrame)
        self.assertIsInstance(cholesky_df.index, pd.DatetimeIndex)
        self.assertListEqual(cholesky_df.columns.to_list(), self.cov.get_names(['AAPL', 'BAC', 'MSFT', 'GOOG']))
        for dt in cholesky_df.index:
            rol_cov = self.rolling_cov.xs(dt, level=0).to_numpy()
            np.testing.assert_array_almost_equal(cholesky[dt][idx], cholesky_df.loc[dt], decimal=16)
            np.testing.assert_array_almost_equal(cholesky[dt], linalg.cholesky(rol_cov, lower=True), decimal=16)
            np.testing.assert_array_almost_equal(cholesky_df.loc[dt], linalg.cholesky(rol_cov, lower=True)[idx],
                                                 decimal=16)

    def test_reverse_cholesky_transformation(self):
        """In this unittest some parts are commented to speed up the process (about 2x), but the numpy and the pandas
        implementations both run and pass - for major changes in class CovarianceHandler it's advised to uncomment
        and tun both implementations"""
        cholesky = self.cov.cholesky_transformation(self.rolling_cov, return_dict=True)
        dict_reversed = self.cov.reverse_cholesky_transformation(cholesky)
        # asset_names = ['AAPL', 'BAC', 'MSFT', 'GOOG']
        cholesky_df = self.cov.cholesky_transformation(self.rolling_cov)
        df_reversed = self.cov.reverse_cholesky_transformation(cholesky_df)
        for dt in dict_reversed.keys():
            rol_cov = self.rolling_cov.xs(dt, level=0)
            np.testing.assert_array_almost_equal(dict_reversed[dt], rol_cov.to_numpy(),
                                                 decimal=16)
            np.testing.assert_array_almost_equal(df_reversed.xs(dt, level=0).to_numpy(), rol_cov.to_numpy(),
                                                 decimal=16)
            # pd.testing.assert_frame_equal(pd.DataFrame(dict_reversed[dt], columns=asset_names, index=asset_names),
            #                               rol_cov)
            # rol_cov.index.names = ['Asset']
            # pd.testing.assert_frame_equal(df_reversed.xs(dt, level=0), rol_cov)

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

    def test_split_names(self):
        tickers = ['AMZN', 'AAPL', 'MSFT', 'GOOG', 'AEE', 'ANSS',
                   'CDNS', 'CSCO', 'CTSH', 'DXC', 'FISV', 'FLT']
        names = self.cov.get_names(tickers)
        self.assertListEqual(tickers, self.cov.split_names(names))
        assets = ['AAPL', 'BAC', 'MSFT', 'GOOG']
        names_assets = self.cov.get_names(assets)
        self.assertListEqual(assets, self.cov.split_names(names_assets))
