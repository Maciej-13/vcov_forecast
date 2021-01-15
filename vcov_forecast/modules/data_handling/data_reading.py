import yfinance as yf

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


class YahooMultipleTickersReader:

    @beartype
    def __init__(self, tickers: (list, str), *args, **kwargs):
        self.__tickers = self.__validate_tickers(tickers)
        self.__data = yf.download(self.__tickers, *args, **kwargs)

    @beartype
    def get_data(self, single_index: bool = False):
        if not single_index:
            return self.__data
        return self.__flatten_index(self.__data)

    def get_columns(self, columns: (list, str), tickers: (list, NoneType) = None, single_index: bool = False):
        columns = [col.title() for col in columns] if isinstance(columns, list) else [columns.title()]
        col_names = list(product(columns, self.__tickers)) if tickers is None else list(product(columns, tickers))
        if not single_index:
            return self.__data.copy()[col_names]
        return self.__flatten_index(self.__data[col_names])

    def get_tickers(self):
        return self.__tickers

    @staticmethod
    def __validate_tickers(tickers):
        if isinstance(tickers, list):
            return tickers
        return tickers.split(' ')

    @staticmethod
    def __flatten_index(df):
        data = df.copy()
        data.columns = [' '.join(col).strip() for col in data.columns]
        return data
