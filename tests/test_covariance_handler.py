import pandas as pd
import numpy as np

from numpy import testing as t
from scipy import linalg

from vcov.modules.data_handling.covariance_handler import CovarianceHandler


def test_covariance_handler():
    cov = CovarianceHandler(15, 4)
    idx = np.tril_indices(4)
    assert cov.lookback == 15
    assert cov.n_assets == 4
    assert all(cov._idx[i][j] == idx[i][j] for i, j in zip(range(len(idx)), range(len(idx))))


def test_calculate_rolling_covariance_matrix(returns, returns_no_idx):
    covariances = CovarianceHandler(15, 4).calculate_rolling_covariance_matrix(returns)
    for i in returns_no_idx.index[14:]:
        temp_data = returns_no_idx.loc[i - 14:i]
        t.assert_array_almost_equal(np.cov(temp_data.to_numpy().T), covariances.loc[
            covariances.index.get_level_values(0).unique()[i - 14]].to_numpy(), decimal=16)


def test_split_covariance_matrices(returns, returns_no_idx):
    dates = returns.index[14:]
    cov = CovarianceHandler(15, 4)
    rolling_cov = cov.calculate_rolling_covariance_matrix(returns)
    cov_by_dt = cov.split_covariance_matrices(rolling_cov)
    for i in range(14, len(returns)):
        temp_data = returns_no_idx.loc[i - 14:i]
        t.assert_array_almost_equal(np.cov(temp_data.to_numpy().T)[np.tril_indices(4)],
                                    cov_by_dt[dates[i - 14]], decimal=16)


def test_split_covariance_to_wide(returns, returns_no_idx):
    cov = CovarianceHandler(15, 4)
    rolling_cov = cov.calculate_rolling_covariance_matrix(returns)
    long_cov = cov.split_covariance_to_wide(rolling_cov)
    for i in range(14, len(returns)):
        cov = np.cov(returns_no_idx.loc[i - 14:i].to_numpy().T)[np.tril_indices(4)].ravel()
        t.assert_array_almost_equal(cov, long_cov.loc[returns.index[i]].to_numpy())


def test_get_covariance_vector(returns, returns_no_idx):
    cov = CovarianceHandler(15, 4)
    rolling_cov = cov.calculate_rolling_covariance_matrix(returns)
    vector = cov.get_covariance_vector(rolling_cov, 'BAC_AAPL')
    filtered_data = returns_no_idx.loc[:, ['AAPL', 'BAC']]
    for i in range(14, len(returns)):
        cov = np.cov(filtered_data.loc[i - 14:i].to_numpy().T)[0][1].ravel()
        t.assert_array_almost_equal(cov, vector.loc[returns.index[i]])


def test_cholesky_transformation(returns):
    cov = CovarianceHandler(15, 4)
    rolling_cov = cov.calculate_rolling_covariance_matrix(returns)
    cholesky = cov.cholesky_transformation(rolling_cov, return_dict=True)
    cholesky_df = cov.cholesky_transformation(rolling_cov)
    idx = np.tril_indices(4)
    assert isinstance(cholesky, dict)
    assert isinstance(cholesky_df, pd.DataFrame)
    assert isinstance(cholesky_df.index, pd.DatetimeIndex)
    assert cholesky_df.columns.to_list() == cov.get_names(['AAPL', 'BAC', 'MSFT', 'GOOG'])
    for dt in cholesky_df.index:
        rol_cov = rolling_cov.xs(dt, level=0).to_numpy()
        t.assert_array_almost_equal(cholesky[dt][idx], cholesky_df.loc[dt], decimal=16)
        t.assert_array_almost_equal(cholesky[dt], linalg.cholesky(rol_cov, lower=True), decimal=16)
        t.assert_array_almost_equal(cholesky_df.loc[dt], linalg.cholesky(rol_cov, lower=True)[idx],
                                    decimal=16)


def test_get_names():
    tickers = ['AMZN', 'AAPL', 'MSFT', 'GOOG', 'AEE', 'ANSS',
               'CDNS', 'CSCO', 'CTSH', 'DXC', 'FISV', 'FLT']
    nms = CovarianceHandler(15, 4).get_names(tickers)
    bench = []
    for nm in tickers:
        for nm_two in tickers:
            if '_'.join((nm, nm_two)) not in bench:
                bench.append('_'.join((nm_two, nm)))

    assert sorted(bench) == sorted(nms)


def test_reverse_cholesky_transformation(returns):
    cov = CovarianceHandler(15, 4)
    rolling_cov = cov.calculate_rolling_covariance_matrix(returns)
    cholesky = cov.cholesky_transformation(rolling_cov, return_dict=True)
    dict_reversed = cov.reverse_cholesky_transformation(cholesky)
    asset_names = ['AAPL', 'BAC', 'MSFT', 'GOOG']
    cholesky_df = cov.cholesky_transformation(rolling_cov)
    df_reversed = cov.reverse_cholesky_transformation(cholesky_df)
    for dt in dict_reversed.keys():
        rol_cov = rolling_cov.xs(dt, level=0)
        t.assert_array_almost_equal(dict_reversed[dt], rol_cov.to_numpy(),
                                    decimal=9)
        t.assert_array_almost_equal(df_reversed.xs(dt, level=0).to_numpy(), rol_cov.to_numpy(),
                                    decimal=9)
        pd.testing.assert_frame_equal(pd.DataFrame(dict_reversed[dt], columns=asset_names, index=asset_names),
                                      rol_cov)
        rol_cov.index.names = ['Asset']
        pd.testing.assert_frame_equal(df_reversed.xs(dt, level=0), rol_cov, check_dtype=False)


def test_split_names():
    tickers = ['AMZN', 'AAPL', 'MSFT', 'GOOG', 'AEE', 'ANSS',
               'CDNS', 'CSCO', 'CTSH', 'DXC', 'FISV', 'FLT']
    names = CovarianceHandler(15, 4).get_names(tickers)
    assert tickers == CovarianceHandler(15, 4).split_names(names)
    assets = ['AAPL', 'BAC', 'MSFT', 'GOOG']
    names_assets = CovarianceHandler(15, 4).get_names(assets)
    assert assets == CovarianceHandler(15, 4).split_names(names_assets)


def test_reverse_cholesky():
    pass
