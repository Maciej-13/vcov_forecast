import yfinance as yf
import pandas as pd

from typing import Union, List
from itertools import product

from pandas.core.frame import DataFrame
from pandas.core.series import Series


class DataReader:

    def __init__(self, tickers: Union[List[str], str], **kwargs) -> None:
        self.tickers = self._validate_tickers(tickers)
        self._data = yf.download(self.tickers, **kwargs)

    def get_data(self, single_index: bool = False) -> Union[DataFrame, Series]:
        if not single_index:
            return self._data.copy()
        return self.__flatten_index(self._data.copy())

    def get_columns(self, columns: Union[List[str], str], single_index: bool = False,
                    tickers: Union[List[str], str, None] = None) -> Union[DataFrame, Series]:
        columns = [col.title() for col in columns] if isinstance(columns, list) else [columns.title()]
        tickers = self._modify_tickers_type(tickers)
        select_columns = [i for i in product(columns, tickers)] if len(self.tickers) != 1 else columns
        if not single_index:
            return self._data.copy()[select_columns]
        return self.__flatten_index(self._data[select_columns])

    def get_data_by_tickers(self, tickers: Union[List[str], str],
                            single_index: bool = False) -> Union[DataFrame, Series]:
        tickers = self._modify_tickers_type(tickers)
        if set(tickers).issubset(self.tickers):
            data = self._data.loc[:, pd.IndexSlice[:, tickers]]
            if single_index:
                data = self.__flatten_index(data)
            return data
        else:
            raise KeyError(f'There is no such ticker as {tickers}. Available tickers are: {self.tickers}')

    def save(self, path: str, single_file: bool = True, columns: Union[List[str], str, None] = None,
             tickers: Union[List[str], str, None] = None, single_index: bool = False) -> None:
        if columns is not None:
            data = self.get_columns(columns=columns, tickers=tickers, single_index=single_index)
        elif tickers is not None:
            data = self.get_data_by_tickers(tickers=tickers, single_index=single_index)
        else:
            data = self.get_data(single_index=single_index)

        tickers = self._modify_tickers_type(tickers)

        if single_file:
            data.to_csv(path + f"/{'_'.join(self.tickers)}.csv")

        elif not single_index:
            for ticker in tickers:
                new_data = data.loc[:, pd.IndexSlice[:, ticker]]
                new_data.to_csv(path + '/' + ticker + '.csv')

        else:
            for ticker in tickers:
                cols = [col for col in data.columns if ticker in col]
                data[cols].to_csv(path + '/' + ticker + '.csv')

    def _modify_tickers_type(self, tickers: Union[List[str], str, None]) -> Union[List[str], str]:
        if tickers is None:
            return self.tickers
        elif isinstance(tickers, str):
            return [tickers]
        else:
            return tickers

    @staticmethod
    def _validate_tickers(tickers: Union[List[str], str]) -> List[str]:
        if isinstance(tickers, list):
            return tickers
        return tickers.split(' ')

    @staticmethod
    def __flatten_index(df: pd.DataFrame) -> DataFrame:
        data = df.copy()
        data.columns = [' '.join(col).strip() for col in data.columns]
        return data
