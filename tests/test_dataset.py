import numpy as np

from keras.preprocessing.sequence import TimeseriesGenerator
from vcov.modules.data_handling.dataset import KerasDataset


def test_keras_dataset(returns):
    dataset = KerasDataset(returns, 15, forward_shift=None)
    assert dataset.data.equals(returns)
    assert dataset.length == 15
    assert dataset.shift is None


def test_get_generator(cov_data):
    generator = KerasDataset(cov_data, forward_shift=None, length=1, batch_size=1).get_generator()
    assert isinstance(generator, TimeseriesGenerator)
    for i, batch in enumerate(generator):
        inputs, target = batch
        np.testing.assert_array_almost_equal(inputs.ravel(), cov_data.iloc[i].to_numpy(), decimal=9)
        np.testing.assert_array_almost_equal(target.ravel(), cov_data.iloc[i + 1].to_numpy(), decimal=9)


def test_get_generator_length(cov_data):
    generator = KerasDataset(cov_data, forward_shift=None, length=5, batch_size=1).get_generator()
    for i, batch in enumerate(generator):
        inputs, target = batch
        np.testing.assert_array_almost_equal(inputs.reshape((5, 10)), cov_data.iloc[i:(i + 5)].to_numpy(), decimal=9)
        np.testing.assert_array_almost_equal(target.ravel(), cov_data.iloc[i + 5].to_numpy(), decimal=9)


def test_get_generator_batch_size(cov_data):
    generator = KerasDataset(cov_data, forward_shift=None, length=1, batch_size=2).get_generator()
    for i, batch in enumerate(generator):
        inputs, target = batch
        i *= 2
        np.testing.assert_array_almost_equal(inputs[0].ravel(), cov_data.iloc[i].to_numpy(), decimal=9)
        np.testing.assert_array_almost_equal(target[0].ravel(), cov_data.iloc[i + 1].to_numpy(), decimal=9)
        np.testing.assert_array_almost_equal(inputs[1].ravel(), cov_data.iloc[i + 1].to_numpy(), decimal=9)
        np.testing.assert_array_almost_equal(target[1].ravel(), cov_data.iloc[i + 2].to_numpy(), decimal=9)


def test_shift_array(cov_data):
    generator = KerasDataset(cov_data, forward_shift=10, length=1, batch_size=1).get_generator()
    assert isinstance(generator, TimeseriesGenerator)
    for i, batch in enumerate(generator):
        inputs, target = batch
        np.testing.assert_array_almost_equal(inputs.ravel(), cov_data.iloc[i].to_numpy(), decimal=9)
        np.testing.assert_array_almost_equal(target.ravel(), cov_data.iloc[i + 10].to_numpy(), decimal=9)


def test_shift_array_length(cov_data):
    generator = KerasDataset(cov_data, forward_shift=10, length=5, batch_size=1).get_generator()
    for i, batch in enumerate(generator):
        inputs, target = batch
        np.testing.assert_array_almost_equal(inputs.reshape((5, 10)), cov_data.iloc[i:(i + 5)].to_numpy(),
                                             decimal=9)
        np.testing.assert_array_almost_equal(target.ravel(), cov_data.iloc[i + 4 + 10].to_numpy(), decimal=9)


def test_shift_array_batch_size(cov_data):
    generator = KerasDataset(cov_data, forward_shift=10, length=1, batch_size=2).get_generator()
    for i, batch in enumerate(generator):
        '''here an additional if statement is required as there are no enough observations to divide such dataset
        into equal batches of two, hence Keras makes the last batch smaller (1 array of 5 observations) instead
        of two arrays'''
        if i < len(generator) - 1:
            inputs, target = batch
            i *= 2
            np.testing.assert_array_almost_equal(inputs[0].ravel(), cov_data.iloc[i].to_numpy(), decimal=9)
            np.testing.assert_array_almost_equal(target[0].ravel(), cov_data.iloc[i + 10].to_numpy(), decimal=9)
            np.testing.assert_array_almost_equal(inputs[1].ravel(), cov_data.iloc[i + 1].to_numpy(), decimal=9)
            np.testing.assert_array_almost_equal(target[1].ravel(), cov_data.iloc[i + 11].to_numpy(), decimal=9)


def test_get_generator_series(cov_data):
    series = cov_data['AAPL_AAPL']
    generator = KerasDataset(series, length=3, batch_size=2).get_generator()
    for i, batch in enumerate(generator):
        inputs, target = batch
        i *= 2
        np.testing.assert_array_almost_equal(inputs[0].ravel(), series.iloc[i:(i + 3)])
        np.testing.assert_array_almost_equal(target[0], series.iloc[i + 3])
        np.testing.assert_array_almost_equal(inputs[1].ravel(), series.iloc[(i + 1):(i + 1 + 3)])
        np.testing.assert_array_almost_equal(target[1], series.iloc[i + 4])
