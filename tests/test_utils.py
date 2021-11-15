import pytest
import numpy as np

from keras.preprocessing.sequence import TimeseriesGenerator

from vcov.modules.utils import get_prepared_generator
from vcov.modules.data_handling.input_handler import InputHandler
from vcov.modules.data_handling.covariance_handler import CovarianceHandler


def test_get_prepared_generator(data_dir):
    with pytest.raises(ValueError) as e:
        get_prepared_generator(data_dir + '/data_short.csv', assets=['AAPL', 'BAC', 'MSFT', 'GOOG'], lookback=15,
                               length=1, val_size=2)
    assert e.value.args[0] == "Parameter val_size must be greater of equal to zero and not higher than 1!"


def test_get_prepared_generator_no_split(data_dir):
    gen = get_prepared_generator(data_dir + '/data_short.csv', assets=['AAPL', 'BAC', 'MSFT', 'GOOG'], lookback=15,
                                 length=1, val_size=0)
    assert isinstance(gen, TimeseriesGenerator)
    assert gen.batch_size == 128
    assert gen.length == 1


def test_get_prepared_generator_batch(data_dir):
    gen = get_prepared_generator(data_dir + '/data_short.csv', assets=['AAPL', 'BAC', 'MSFT', 'GOOG'], lookback=15,
                                 length=1, val_size=0, batch_size=1)
    assert isinstance(gen, TimeseriesGenerator)
    assert gen.batch_size == 1

    data = InputHandler(data_dir+'/data_short.csv', assets=['AAPL', 'BAC', 'MSFT', 'GOOG'], column='Close',
                        returns=True).get_data()
    cov = CovarianceHandler(lookback=15, n_assets=4)
    chol_data = cov.cholesky_transformation(cov.calculate_rolling_covariance_matrix(data))
    for i, batch in enumerate(gen):
        inputs, target = batch
        np.testing.assert_array_almost_equal(inputs.reshape((10,)), chol_data.iloc[i].to_numpy(),
                                             decimal=8)
        np.testing.assert_array_almost_equal(target.ravel(), chol_data.iloc[i + 1].to_numpy(), decimal=8)


def test_get_prepared_generator_split(data_dir):
    train_gen, val_gen = get_prepared_generator(data_dir + '/data_short.csv', assets=['AAPL', 'BAC', 'MSFT', 'GOOG'],
                                                lookback=15, length=1, val_size=0.2, batch_size=1)
    assert isinstance(train_gen, TimeseriesGenerator)
    assert isinstance(val_gen, TimeseriesGenerator)
    train_data, val_data = InputHandler(data_dir + '/data_short.csv', assets=['AAPL', 'BAC', 'MSFT', 'GOOG'],
                                        column='Close', returns=True).train_test_split(0.2)
    cov = CovarianceHandler(lookback=15, n_assets=4)
    train_data = cov.cholesky_transformation(cov.calculate_rolling_covariance_matrix(train_data))
    val_data = cov.cholesky_transformation(cov.calculate_rolling_covariance_matrix(val_data))

    for i, batch in enumerate(train_gen):
        inputs, target = batch
        np.testing.assert_array_almost_equal(inputs.reshape((10,)), train_data.iloc[i].to_numpy())
        np.testing.assert_array_almost_equal(target.ravel(), train_data.iloc[i + 1].to_numpy())

    for i, batch in enumerate(val_gen):
        inputs, target = batch
        np.testing.assert_array_almost_equal(inputs.reshape((10,)), val_data.iloc[i].to_numpy())
        np.testing.assert_array_almost_equal(target.ravel(), val_data.iloc[i + 1].to_numpy())
