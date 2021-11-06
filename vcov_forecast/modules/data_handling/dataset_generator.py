import pandas as pd
import numpy as np

from collections import OrderedDict
from sklearn.model_selection import train_test_split
from keras.preprocessing.sequence import TimeseriesGenerator
from beartype import beartype
from beartype.cave import NoneType


class InputHandler:

    @beartype
    def __init__(self, path: str, assets: list, column: str = 'Close', returns: bool = True):
        self.__path = path
        self.assets = assets
        self.__column = column
        self.__load_data()
        self.__select_column()
        if returns:
            self.__calculate_returns()

    def get_data(self):
        return self.__data

    def train_test_split(self, test_size: float = 0.2):
        return train_test_split(self.__data, test_size=test_size, shuffle=False, random_state=42)

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

    @beartype
    def __init__(self, lookback: int, n_assets: int):
        self.lookback = lookback
        self.__assets = n_assets
        self.__idx = np.tril_indices(self.__assets)

    @beartype
    def calculate_rolling_covariance_matrix(self, data: pd.DataFrame):
        data = data.rolling(self.lookback).cov()
        data.dropna(inplace=True)
        return data

    @beartype
    def split_covariance_matrices(self, rolling_mtx: pd.DataFrame):
        cov_by_date = {dt: np.array(rolling_mtx.xs(dt, level=0))[self.__idx] for dt in
                       rolling_mtx.index.get_level_values(0).unique()}
        return cov_by_date

    @beartype
    def split_covariance_to_wide(self, rolling_mtx: pd.DataFrame):
        dates = rolling_mtx.index.get_level_values(0).unique()
        data = pd.DataFrame(columns=self.get_names(rolling_mtx.columns.tolist()), index=dates)
        for dt in dates:
            data.loc[dt] = rolling_mtx.xs(dt, level=0).to_numpy()[self.__idx]
        return data

    @beartype
    def get_covariance_vector(self, rolling_mtx: pd.DataFrame, name: str):
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

    @beartype
    def cholesky_transformation(self, rolling_mtx: pd.DataFrame, return_dict: bool = False):
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

    @beartype
    def reverse_cholesky_transformation(self, cholesky: (dict, pd.DataFrame)):
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
            return data.astype(np.float32)

        else:
            raise TypeError(f'Object of type {type(cholesky)} is not supported! Pass a dictionary or a data frame!')

    @staticmethod
    @beartype
    def get_names(ticker_list: list):
        first, second = np.tril_indices(len(ticker_list))
        return ['_'.join((ticker_list[i], ticker_list[j])) for i, j in zip(first, second)]

    @staticmethod
    @beartype
    def split_names(joined_names: list):
        return list(OrderedDict.fromkeys([name.split('_')[0] for name in joined_names]))

    @staticmethod
    def __reverse_cholesky(matrix):
        return np.dot(matrix, matrix.T.conj())


class KerasDataset:

    @beartype
    def __init__(self, data: (pd.DataFrame, pd.Series), length: int, forward_shift: (int, NoneType) = None, **kwargs):
        self.__data = data
        self.__shift = forward_shift if forward_shift is None else forward_shift - 1 if forward_shift > 1 else 0
        self.__length = length
        self.__generator = None
        self.__create_generator(**kwargs)

    def get_generator(self):
        return self.__generator

    def __create_generator(self, **kwargs):
        if self.__shift is None:
            data = self.__prepare_arrays()
            self.__generator = TimeseriesGenerator(data, data, length=self.__length, **kwargs)

        else:
            data = self.__prepare_arrays()
            data, targets = self.__shift_array(data)
            self.__generator = TimeseriesGenerator(data, targets, length=self.__length, **kwargs)

    def __prepare_arrays(self):
        if isinstance(self.__data, pd.Series):
            return np.array(self.__data).astype(np.float32).reshape((len(self.__data), 1))

        else:
            arrays = tuple(np.array(self.__data[i]).reshape((len(self.__data), 1)) for i in self.__data.columns)
            matrix = np.hstack(arrays).astype(np.float32)
            return matrix

    def __shift_array(self, array):
        target = array[self.__shift:]
        values = array[:(len(array) - self.__shift)]
        return values.astype(np.float32), target.astype(np.float32)


@beartype
def get_prepared_generator(path: str, assets: list, lookback: int, length: int, forward_shift: (int, NoneType) = None,
                           val_size: (float, int) = 0.2, **kwargs):
    if 0 < val_size < 1:
        train, val = InputHandler(path, assets).train_test_split(val_size)
        cov_handler = CovarianceHandler(lookback, n_assets=len(assets))
        train_cholesky = cov_handler.cholesky_transformation(cov_handler.calculate_rolling_covariance_matrix(train))
        val_cholesky = cov_handler.cholesky_transformation(cov_handler.calculate_rolling_covariance_matrix(val))
        train_gen = KerasDataset(train_cholesky, length=length, forward_shift=forward_shift, **kwargs).get_generator()
        val_gen = KerasDataset(val_cholesky, length=length, forward_shift=forward_shift, **kwargs).get_generator()
        return train_gen, val_gen

    elif val_size == 0:
        data = InputHandler(path, assets).get_data()
        cov_handler = CovarianceHandler(lookback, n_assets=len(assets))
        cholesky_data = cov_handler.cholesky_transformation(cov_handler.calculate_rolling_covariance_matrix(data))
        gen = KerasDataset(cholesky_data, length=length, forward_shift=forward_shift, **kwargs).get_generator()
        return gen

    else:
        raise (ValueError("Parameter val_size must be greater of equal to zero and not higher than 1!"))
