import tensorflow as tf
from tensorflow import keras
from tcn.tcn import TCN

from vcov.modules.models.lstm_model import LSTM, KerasTCN


def test_lstm():
    nn = LSTM(shape=(5, 10), architecture=[5, 5, 5], dropout_rate=0.1)
    assert isinstance(nn, LSTM)
    assert len(getattr(nn, '_model').layers) == (1 + 3 * 2 + 1)


def test_compile_model(lstm):
    lstm.compile_model(loss=tf.losses.mean_squared_error, metrics=[tf.metrics.mean_absolute_error],
                       optimizer=tf.keras.optimizers.Adam())
    model = getattr(lstm, '_model')
    assert isinstance(model.compiled_loss._loss_metric, tf.keras.metrics.Mean)
    assert model._is_compiled
    assert isinstance(model.optimizer, tf.keras.optimizers.Adam)


def test_get_model(lstm):
    model = lstm.get_model()
    assert isinstance(model, keras.Model)
    assert model.built
    assert len(model.layers) == 8
    assert isinstance(model.layers[0], keras.layers.InputLayer)
    for i in range(1, 6, 2):
        assert isinstance(model.layers[i], keras.layers.Dropout)
        assert isinstance(model.layers[i + 1], keras.layers.LSTM)
    assert isinstance(model.layers[-1], keras.layers.Dense)


def test_get_model_conv_lstm():
    conv_lstm = LSTM(shape=(5, 10), architecture=[5, 5, 5], dropout_rate=0.1, convolutional_layer=True,
                     kernel_size=1, padding='same', filters=64, pool_size=2)
    model = conv_lstm.get_model()
    assert isinstance(model, keras.Model)
    assert model.built
    assert len(model.layers) == 11
    assert isinstance(model.layers[0], keras.layers.InputLayer)
    assert isinstance(model.layers[1], keras.layers.Conv1D)
    assert isinstance(model.layers[2], keras.layers.MaxPool1D)
    assert isinstance(model.layers[3], keras.layers.TimeDistributed)
    for i in range(4, 9, 2):
        assert isinstance(model.layers[i], keras.layers.Dropout)
        assert isinstance(model.layers[i + 1], keras.layers.LSTM)
    assert isinstance(model.layers[-1], keras.layers.Dense)


def test_get_model_bi_lstm():
    bi_lstm = LSTM(shape=(5, 10), architecture=[5, 5, 5], dropout_rate=0.1, bidirectional=True,
                   merge_mode='concat')
    model = bi_lstm.get_model()
    assert isinstance(model, keras.Model)
    assert model.built
    assert len(model.layers) == 8
    assert isinstance(model.layers[0], keras.layers.InputLayer)
    for i in range(1, 5, 2):
        assert isinstance(model.layers[i], keras.layers.Dropout)
        assert isinstance(model.layers[i + 1], keras.layers.Bidirectional)
    assert isinstance(model.layers[-3], keras.layers.Dropout)
    assert isinstance(model.layers[-2], keras.layers.LSTM)
    assert isinstance(model.layers[-1], keras.layers.Dense)


def test_get_model_conv_bi_lstm():
    conv_bi_lstm = LSTM(shape=(5, 10), architecture=[5, 5, 5], dropout_rate=0.1, bidirectional=True,
                        merge_mode='concat', convolutional_layer=True, kernel_size=1, padding='same',
                        filters=64, pool_size=2)
    model = conv_bi_lstm.get_model()
    assert isinstance(model, keras.Model)
    assert model.built
    assert len(model.layers) == 11
    assert isinstance(model.layers[0], keras.layers.InputLayer)
    assert isinstance(model.layers[1], keras.layers.Conv1D)
    assert isinstance(model.layers[2], keras.layers.MaxPool1D)
    assert isinstance(model.layers[3], keras.layers.TimeDistributed)
    for i in range(4, 7, 2):
        assert isinstance(model.layers[i], keras.layers.Dropout)
        assert isinstance(model.layers[i + 1], keras.layers.Bidirectional)
    assert isinstance(model.layers[-3], keras.layers.Dropout)
    assert isinstance(model.layers[-2], keras.layers.LSTM)
    assert isinstance(model.layers[-1], keras.layers.Dense)


def test_get_summary(capsys, lstm):
    lstm.get_summary()
    captured = capsys.readouterr()
    assert captured


def test_tcn():
    tcn = KerasTCN(shape=(5, 5), nb_filters=25, kernel_size=3, nb_stacks=1, dilations=(1, 2), dropout=0.2,
                   padding="same")
    assert hasattr(tcn, '_model')
    model = getattr(tcn, '_model')
    assert len(model.layers) == 2
    assert isinstance(model.layers[-1], keras.layers.Dense)


def test_tcn_compile_model(tcn):
    tcn.compile_model(loss=tf.losses.mean_squared_error, metrics=[tf.metrics.mean_absolute_error],
                      optimizer=tf.keras.optimizers.Adam())
    model = getattr(tcn, '_model')
    assert isinstance(model.compiled_loss._loss_metric, tf.keras.metrics.Mean)
    assert model._is_compiled
    assert isinstance(model.optimizer, tf.keras.optimizers.Adam)


def test_tcn_get_model(tcn):
    model = tcn.get_model()
    assert isinstance(model, keras.Model)
    assert model.built
    assert len(model.layers) == 2
    assert isinstance(model.layers[0], TCN)
    assert isinstance(model.layers[1], keras.layers.Dense)


def test_tcn_get_summary(capsys, tcn):
    tcn.get_summary()
    captured = capsys.readouterr()
    assert captured
