import yfinance as yf
import pandas as pd

from warnings import warn
from beartype import beartype
from beartype.cave import NoneType
from itertools import product


class YahooDataReader:

    @beartype
    def __init__(self, ticker: str, *args, **kwargs):
        self.__ticker = ticker
        self.__is_ticker_valid()
        self.__data = yf.download(self.__ticker, *args, **kwargs)

    def get_open(self):
        return self.__data["Open"]

    def get_close(self):
        return self.__data["Close"]

    def get_adj_close(self):
        return self.__data["Adj Close"]

    def get_volume(self):
        return self.__data["Volume"]

    def get_low(self):
        return self.__data["Low"]

    def get_high(self):
        return self.__data["High"]

    def get_data(self):
        return self.__data

    def get_ticker(self):
        return self.__ticker

    def __is_ticker_valid(self):
        if len(self.__ticker.split(' ')) > 1:
            warn(f'{len(self.__ticker.split(" "))} tickers provided, but '
                 f'only {self.__ticker.split(" ")[0]} will be used')
            self.__ticker = self.__ticker.split(" ")[0]


class YahooReader:

    @beartype
    def __init__(self, tickers: (list, str), *args, **kwargs):
        self.__tickers = self.__validate_tickers(tickers)
        self.__data = yf.download(self.__tickers, *args, **kwargs)

    @beartype
    def get_data(self, single_index: bool = False):
        if not single_index:
            return self.__data.copy()
        return self.__flatten_index(self.__data.copy())

    @beartype
    def get_columns(self, columns: (list, str), tickers: (list, str, NoneType) = None, single_index: bool = False):
        columns = [col.title() for col in columns] if isinstance(columns, list) else [columns.title()]
        tickers = self.__check_tickers_type(tickers)
        col_names = list(product(columns, tickers))
        if not single_index:
            return self.__data.copy()[col_names]
        else:
            return self.__flatten_index(self.__data[col_names])

    def get_all_tickers(self):
        return self.__tickers

    @beartype
    def get_data_by_tickers(self, tickers: (str, list), single_index: bool = False):
        tickers = self.__check_tickers_type(tickers)
        if set(tickers).issubset(self.__tickers):
            data = self.__data.loc[:, pd.IndexSlice[:, tickers]]
            if single_index:
                data = self.__flatten_index(data)
            return data
        else:
            raise KeyError(f'There is no such ticker as {tickers}. Available tickers are: {self.get_all_tickers()}')

    @beartype
    def save(self, path: str, single_file: bool = True, columns: (list, str, NoneType) = None,
             tickers: (list, str, NoneType) = None, single_index: bool = False):
        if columns is not None:
            data = self.get_columns(columns=columns, tickers=tickers, single_index=single_index)
        elif tickers is not None:
            data = self.get_data_by_tickers(tickers=tickers, single_index=single_index)
        else:
            data = self.get_data(single_index=single_index)

        tickers = self.__check_tickers_type(tickers)

        if single_file:
            data.to_csv(path)

        elif not single_index:
            for ticker in tickers:
                new_data = data.loc[:, pd.IndexSlice[:, ticker]]
                new_data.to_csv(path + '/' + ticker + '.csv')

        else:
            for ticker in tickers:
                cols = [col for col in data.columns if ticker in col]
                data[cols].to_csv(path + '/' + ticker + '.csv')

    @staticmethod
    @beartype
    def __validate_tickers(tickers: (list, str)):
        if isinstance(tickers, list):
            return tickers
        return tickers.split(' ')

    @staticmethod
    @beartype
    def __flatten_index(df: pd.DataFrame):
        data = df.copy()
        data.columns = [' '.join(col).strip() for col in data.columns]
        return data

    @beartype
    def __check_tickers_type(self, tickers: (list, str, NoneType)):
        if tickers is None:
            return self.__tickers
        elif isinstance(tickers, str):
            return [tickers]
        else:
            return tickers

