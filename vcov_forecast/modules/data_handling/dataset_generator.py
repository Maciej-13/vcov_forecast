import pandas as pd
import numpy as np

from collections import OrderedDict


class InputHandler:

    def __init__(self, path, assets, column='Close', returns=True):
        self.__path = path
        self.assets = assets
        self.__column = column
        self.__load_data()
        self.__select_column()
        if returns:
            self.__calculate_returns()

    def get_data(self):
        return self.__data

    def __load_data(self):
        self.__data = pd.read_csv(self.__path, parse_dates=['Date'], index_col='Date')

    def __select_column(self):
        if self.__column is not None:
            close_col = [self.__column + ' ' + a for a in self.assets]
            self.__data = self.__data[close_col]
            self.__data.columns = [c.split()[-1].strip() for c in self.__data.columns]

    def __calculate_returns(self):
        self.__data = self.__data.pct_change(1)
        self.__data.dropna(inplace=True)


class CovarianceHandler:

    def __init__(self, lookback, n_assets):
        self.lookback = lookback
        self.__assets = n_assets
        self.__idx = np.tril_indices(self.__assets)

    def calculate_rolling_covariance_matrix(self, data):
        data = data.rolling(self.lookback).cov()
        data.dropna(inplace=True)
        return data

    def split_covariance_matrices(self, rolling_mtx):
        cov_by_date = {dt: np.array(rolling_mtx.xs(dt, level=0))[self.__idx] for dt in
                       rolling_mtx.index.get_level_values(0).unique()}
        return cov_by_date

    def split_covariance_to_wide(self, rolling_mtx):
        dates = rolling_mtx.index.get_level_values(0).unique()
        data = pd.DataFrame(columns=self.get_names(rolling_mtx.columns.tolist()), index=dates)
        for dt in dates:
            data.loc[dt] = rolling_mtx.xs(dt, level=0).to_numpy()[self.__idx]
        return data

    def get_covariance_vector(self, rolling_mtx, name):
        cov_names = self.get_names(rolling_mtx.columns.to_list())
        if name not in cov_names:
            reversed_name = '_'.join((name.split('_')[1], name.split('_')[0]))
            if reversed_name in cov_names:
                name = reversed_name
            else:
                raise ValueError(f'{name} if not a valid pair of tickers! Available pairs are {cov_names}')
        row, col = name.split('_')
        filtered_df = rolling_mtx.copy()
        filtered_df = filtered_df.xs(row, level=1)[col]
        return filtered_df

    def cholesky_transformation(self, rolling_mtx, return_dict=False):
        if return_dict:
            return {dt: np.linalg.cholesky(rolling_mtx.xs(dt, level=0).to_numpy()) for dt in
                    rolling_mtx.index.get_level_values(0)}
        else:
            dates = rolling_mtx.index.get_level_values(0).unique()
            data = pd.DataFrame(columns=self.get_names(rolling_mtx.columns.tolist()),
                                index=dates)
            for dt in dates:
                data.loc[dt] = np.linalg.cholesky(rolling_mtx.xs(dt, level=0).to_numpy())[self.__idx]
            return data

    def reverse_cholesky_transformation(self, cholesky):
        if isinstance(cholesky, dict):
            return {dt: self.__reverse_cholesky(matrix) for dt, matrix in cholesky.items()}

        elif isinstance(cholesky, pd.DataFrame):
            assets = self.split_names(cholesky.columns.tolist())
            arrays = [np.repeat(cholesky.index.tolist(), len(assets)), assets * len(cholesky.index)]
            tuples = list(zip(*arrays))
            m_index = pd.MultiIndex.from_tuples(tuples, names=['Date', 'Asset'])
            data = pd.DataFrame(columns=assets, index=m_index)
            for dt in cholesky.index:
                temp_matrix = np.zeros((self.__assets, self.__assets)).astype(float)
                temp_matrix[self.__idx] = cholesky.loc[dt, :].values
                data.loc[(dt, slice(None)), :] = self.__reverse_cholesky(temp_matrix)
            return data.astype(float)

        else:
            raise TypeError(f'Object of type {type(cholesky)} is not supported! Pass a dictionary or a data frame!')

    @staticmethod
    def get_names(ticker_list):
        first, second = np.tril_indices(len(ticker_list))
        return ['_'.join((ticker_list[i], ticker_list[j])) for i, j in zip(first, second)]

    @staticmethod
    def split_names(joined_names):
        return list(OrderedDict.fromkeys([name.split('_')[0] for name in joined_names]))

    @staticmethod
    def __reverse_cholesky(matrix):
        return np.dot(matrix, matrix.T.conj())
