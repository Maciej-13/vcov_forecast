import pandas as pd
import numpy as np


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

    def calculate_rolling_covariance_matrix(self, data):
        data = data.rolling(self.lookback).cov()
        data.dropna(inplace=True)
        return data

    def split_covariance_matrices(self, rolling_mtx):
        idx = np.tril_indices(self.__assets)
        cov_by_date = {dt: np.array(rolling_mtx.xs(dt, level=0))[idx] for dt in
                       rolling_mtx.index.get_level_values(0)}
        return cov_by_date

    def split_covariance_to_long(self, rolling_mtx):
        dates = rolling_mtx.index.get_level_values(0).unique()
        data = pd.DataFrame(columns=self.get_names(rolling_mtx.columns.tolist()), index=dates)
        idx = np.tril_indices(self.__assets)
        for dt in dates:
            data.loc[dt] = rolling_mtx.xs(dt, level=0).to_numpy()[idx]
        return data

    @staticmethod
    def get_names(ticker_list):
        first, second = np.tril_indices(len(ticker_list))
        return ['_'.join((ticker_list[i], ticker_list[j])) for i, j in zip(first, second)]
