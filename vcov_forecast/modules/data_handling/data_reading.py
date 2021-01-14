import yfinance as yf

from warnings import warn
from beartype import beartype


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
