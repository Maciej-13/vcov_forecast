import numpy as np
import pandas as pd

from vcov.modules.strategy.performance import PerformanceStatistics


def test_annualized_returns(single_prices):
    s1 = 365
    s2 = 252
    r = single_prices.iloc[-1] / single_prices.iloc[0]
    assert r ** (s1 / len(single_prices)) - 1 == PerformanceStatistics.annualized_returns(single_prices, s1)
    assert r ** (s2 / len(single_prices)) - 1 == PerformanceStatistics.annualized_returns(single_prices, s2)


def test_annualized_standard_deviation(single_prices):
    s1 = 365
    s2 = 252
    r = single_prices.diff(1) / single_prices.shift(1)
    assert round(np.std(r) * np.sqrt(s1), 10) == round(
        PerformanceStatistics().annualized_standard_deviation(single_prices, s1), 10)
    assert round(np.std(r) * np.sqrt(s2), 10) == round(
        PerformanceStatistics().annualized_standard_deviation(single_prices, s2), 10)


def test_maximum_daily_drawdown(single_prices):
    mdd = (single_prices / single_prices.rolling(len(single_prices), min_periods=1).max() - 1).rolling(
        len(single_prices), min_periods=1).min()
    assert mdd.equals(PerformanceStatistics.maximum_daily_drawdown(single_prices))


def test_maximum_drawdown(single_prices):
    mdd = (single_prices / single_prices.rolling(len(single_prices), min_periods=1).max() - 1).rolling(
        len(single_prices), min_periods=1).min()
    assert min(mdd) == PerformanceStatistics().maximum_drawdown(single_prices)


def test_information_ratio(single_prices):
    s1 = 365
    s2 = 252
    ar = single_prices.iloc[-1] / single_prices.iloc[0]
    r = single_prices.diff(1) / single_prices.shift(1)
    assert round((ar ** (s1 / len(single_prices)) - 1) / (np.std(r) * np.sqrt(s1)), 10) == round(
        PerformanceStatistics().information_ratio(single_prices, s1), 10)
    assert round((ar ** (s2 / len(single_prices)) - 1) / (np.std(r) * np.sqrt(s2)), 10) == round(
        PerformanceStatistics().information_ratio(single_prices, s2), 10)


def test_loss_duration(single_prices):
    s1 = 365
    s2 = 252
    ld1 = PerformanceStatistics.loss_duration(single_prices, s1)
    ld2 = PerformanceStatistics.loss_duration(single_prices, s2)
    assert isinstance(ld1, pd.Series)
    assert isinstance(ld2, pd.Series)
    assert isinstance(ld1.index, pd.DatetimeIndex)
    assert isinstance(ld2.index, pd.DatetimeIndex)
    assert len(ld1) == len(ld2) == len(single_prices)


def test_loss_duration_artificial():
    prices = pd.Series([1, 1, 0, 1, 2, 3, 2.5, 2, 1.5, 1, 0.5, 0, 1, 3, 1.5, 1, 0.5, 0, 1, 1],
                       index=pd.date_range("01-01-2021", "01-20-2021"))
    s = 1
    ld = PerformanceStatistics.loss_duration(prices, s)
    assert ld.to_list() == [0, 0, 1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6]


def test_max_loss_duration():
    prices = pd.Series([1, 1, 0, 1, 2, 3, 2.5, 2, 1.5, 1, 0.5, 0, 1, 3, 1.5, 1, 0.5, 0, 1, 1],
                       index=pd.date_range("01-01-2021", "01-20-2021"))
    s = 1
    ld = PerformanceStatistics().max_loss_duration(prices, s)
    assert ld == 7


def test_calculate_r(single_prices):
    pd.testing.assert_series_equal(PerformanceStatistics._calculate_r(single_prices),
                                   single_prices.diff(1) / single_prices.shift(1))


def test_get_performance_statistics(single_prices):
    s = 365
    stats = PerformanceStatistics().get_performance_statistics(single_prices, scale=s)
    ret = single_prices.iloc[-1] / single_prices.iloc[0]
    arc = ret ** (s / len(single_prices)) - 1
    r = single_prices.diff(1) / single_prices.shift(1)
    asd = np.std(r) * np.sqrt(s)
    ir = arc / asd
    md = min((single_prices / single_prices.rolling(len(single_prices), min_periods=1).max() - 1).rolling(
        len(single_prices), min_periods=1).min())
    mld = PerformanceStatistics().max_loss_duration(single_prices, s)
    assert isinstance(stats, dict)
    assert 'aRC' in stats
    assert isinstance(stats['aRC'], float)
    assert 'aSD' in stats
    assert isinstance(stats['aRC'], float)
    assert 'MD' in stats
    assert isinstance(stats['aRC'], float)
    assert 'MLD' in stats
    assert isinstance(stats['aRC'], float)
    assert 'IR' in stats
    assert isinstance(stats['aRC'], float)
    assert 'IR2' in stats
    assert isinstance(stats['aRC'], float)
    assert round(stats['IR2'], 10) == round(ir * arc * np.sign(arc) / md, 10)
    assert 'IR3' in stats
    assert isinstance(stats['aRC'], float)
    assert round(stats['IR3'], 10) == round((arc ** 3) / (asd * md * mld), 10)


def test_cumulative_returns(single_prices):
    cum_ret = PerformanceStatistics.cumulative_returns(single_prices)
    assert isinstance(cum_ret, pd.Series)
    assert all(
        i == j for i, j in zip(cum_ret.dropna(), (np.cumprod(single_prices / single_prices.shift(1)) - 1).dropna()))
