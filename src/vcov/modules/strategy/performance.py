import numpy as np
import pandas as pd

from typing import Dict, List
from pandas.core.series import Series


class PerformanceStatistics:

    def get_performance_statistics(self, prices: Series, scale: int = 365) -> Dict[str, float]:
        arc = self.annualized_returns(prices, scale)
        asd = self.annualized_standard_deviation(prices, scale)
        ir = self.information_ratio(prices, scale)
        md = self.maximum_drawdown(prices)
        mld = self.max_loss_duration(prices, scale)
        return {'aRC': arc, 'aSD': asd, 'MD': md, 'MLD': mld, 'IR': ir, 'IR2': ir * arc * np.sign(arc) / md,
                'IR3': (arc ** 3) / (asd * md * mld)}

    @staticmethod
    def annualized_returns(prices: Series, scale: int = 365) -> float:
        r: float = (prices.iloc[-1] / prices.iloc[0])
        return r ** (scale / len(prices)) - 1

    def annualized_standard_deviation(self, prices: Series, scale: int = 365) -> float:
        r: Series = self._calculate_r(prices)
        asd: float = np.std(r) * np.sqrt(scale)
        return asd

    @staticmethod
    def maximum_daily_drawdown(prices: Series) -> Series:
        return (prices / prices.cummax() - 1).cummin()

    def maximum_drawdown(self, prices) -> float:
        md: float = min(self.maximum_daily_drawdown(prices))
        return md

    def information_ratio(self, prices: Series, scale: int = 365) -> float:
        return self.annualized_returns(prices, scale) / self.annualized_standard_deviation(prices, scale)

    @staticmethod
    def loss_duration(prices: Series, scale: int = 365) -> Series:
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
        ld: Series = self.loss_duration(prices, scale)
        mld: float = max(ld)
        return mld

    @staticmethod
    def _calculate_r(prices: Series) -> Series:
        return prices / prices.shift(1)
