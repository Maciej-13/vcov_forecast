import numpy as np
import pandas as pd

from typing import Dict, List
from pandas.core.series import Series


class PerformanceStatistics:

    def get_performance_statistics(self, prices: Series, scale: int = 365) -> Dict[str, float]:
        prices = prices.dropna()
        arc = self.annualized_returns(prices, scale)
        asd = self.annualized_standard_deviation(prices, scale)
        ir = self.information_ratio(prices, scale)
        md = self.maximum_drawdown(prices)
        mld = self.max_loss_duration(prices, scale)
        return {'aRC': arc, 'aSD': asd, 'MD': md, 'MLD': mld, 'IR': ir, 'IR2': ir * arc * np.sign(arc) / md,
                'IR3': (arc ** 3) / (asd * md * mld)}

    @staticmethod
    def annualized_returns(prices: Series, scale: int = 365) -> float:
        r: float = (prices.dropna().iloc[-1] / prices.dropna().iloc[0])
        return r ** (scale / len(prices)) - 1

    def annualized_standard_deviation(self, prices: Series, scale: int = 365) -> float:
        r: Series = self._calculate_r(prices.dropna())
        asd: float = np.std(r) * np.sqrt(scale)
        return asd

    @staticmethod
    def maximum_daily_drawdown(prices: Series) -> Series:
        prices = prices.dropna()
        return ((prices.cummax() - prices) / prices.cummax()).cummax()

    def maximum_drawdown(self, prices) -> float:
        md: float = max(self.maximum_daily_drawdown(prices.dropna()))
        return md

    def information_ratio(self, prices: Series, scale: int = 365) -> float:
        return self.annualized_returns(prices.dropna(), scale) / self.annualized_standard_deviation(prices.dropna(),
                                                                                                    scale)

    @staticmethod
    def loss_duration(prices: Series, scale: int = 365) -> Series:
        prices = prices.dropna()
        current: float = prices.iloc[0]
        ld: List[int] = []
        for i, p in enumerate(prices):
            if p >= current:
                current = p
                ld.append(0)
            else:
                ld.append(ld[i - 1] + 1)

        return pd.Series(ld, index=prices.index) / scale

    def max_loss_duration(self, prices: Series, scale: int = 365) -> float:
        ld: Series = self.loss_duration(prices.dropna(), scale)
        mld: float = max(ld)
        return mld

    @staticmethod
    def cumulative_returns(prices: Series) -> Series:
        returns: Series = prices.dropna().pct_change()
        return (returns + 1).cumprod() - 1

    @staticmethod
    def _calculate_r(prices: Series) -> Series:
        r: Series = prices.dropna() / prices.dropna().shift(1) - 1
        r = r.replace([np.inf, -np.inf], np.nan)
        return r
