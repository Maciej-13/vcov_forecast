import pandas as pd
import numpy as np

from typing import Union, Dict, List
from collections import OrderedDict
from pandas.core.frame import DataFrame


class CovarianceHandler:

    def __init__(self, lookback: int, n_assets: int):
        self.lookback = lookback
        self.n_assets = n_assets
        self._idx = np.tril_indices(self.n_assets)

    def calculate_rolling_covariance_matrix(self, data: pd.DataFrame) -> DataFrame:
        data = data.rolling(self.lookback).cov()
        data.dropna(inplace=True)
        return data

    def split_covariance_matrices(self, rolling_mtx: pd.DataFrame) -> Dict[pd.Timestamp, np.ndarray]:
        cov_by_date = {dt: np.array(rolling_mtx.xs(dt, level=0))[self._idx] for dt in
                       rolling_mtx.index.get_level_values(0).unique()}
        return cov_by_date

    def split_covariance_to_wide(self, rolling_mtx: pd.DataFrame) -> DataFrame:
        dates = rolling_mtx.index.get_level_values(0).unique()
        data = pd.DataFrame(columns=self.get_names(rolling_mtx.columns.tolist()), index=dates)
        for dt in dates:
            data.loc[dt] = rolling_mtx.xs(dt, level=0).to_numpy()[self._idx]
        return data

    def get_covariance_vector(self, rolling_mtx: pd.DataFrame, name: str) -> DataFrame:
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

    def cholesky_transformation(self, rolling_mtx: pd.DataFrame,
                                return_dict: bool = False) -> Union[Dict[pd.Timestamp, np.ndarray], DataFrame]:
        if return_dict:
            return {dt: np.linalg.cholesky(rolling_mtx.xs(dt, level=0).to_numpy()) for dt in
                    rolling_mtx.index.get_level_values(0)}
        else:
            dates = rolling_mtx.index.get_level_values(0).unique()
            data = pd.DataFrame(columns=self.get_names(rolling_mtx.columns.tolist()),
                                index=dates)
            for dt in dates:
                data.loc[dt] = np.linalg.cholesky(rolling_mtx.xs(dt, level=0).to_numpy())[self._idx]
            return data

    def reverse_cholesky_transformation(self, cholesky: Union[Dict[pd.Timestamp, np.ndarray], pd.DataFrame]
                                        ) -> Union[Dict[pd.Timestamp, np.ndarray], DataFrame]:
        if isinstance(cholesky, dict):
            return {dt: self.reverse_cholesky(matrix) for dt, matrix in cholesky.items()}

        elif isinstance(cholesky, pd.DataFrame):
            assets = self.split_names(cholesky.columns.tolist())
            arrays = [np.repeat(cholesky.index.tolist(), len(assets)), assets * len(cholesky.index)]
            tuples = list(zip(*arrays))
            m_index = pd.MultiIndex.from_tuples(tuples, names=['Date', 'Asset'])
            data = pd.DataFrame(columns=assets, index=m_index)
            for dt in cholesky.index:
                temp_matrix = np.zeros((self.n_assets, self.n_assets)).astype(float)
                temp_matrix[self._idx] = cholesky.loc[dt, :].values
                data.loc[(dt, slice(None)), :] = self.reverse_cholesky(temp_matrix)
            return data.astype(np.float32)

        else:
            raise TypeError(f'Object of type {type(cholesky)} is not supported! Pass a dictionary or a data frame!')

    @staticmethod
    def get_names(ticker_list: List[str]) -> List[str]:
        first, second = np.tril_indices(len(ticker_list))
        return ['_'.join((ticker_list[i], ticker_list[j])) for i, j in zip(first, second)]

    @staticmethod
    def split_names(joined_names: List[str]) -> List[str]:
        return list(OrderedDict.fromkeys([name.split('_')[0] for name in joined_names]))

    @staticmethod
    def reverse_cholesky(matrix: np.ndarray) -> np.ndarray:
        return np.dot(matrix, matrix.T.conj())
