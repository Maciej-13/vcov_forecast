import unittest
from unittest.mock import patch

import tensorflow as tf
from tensorflow import keras

from modules.models.keras_model import LSTM


class TestLSTM(unittest.TestCase):

    def setUp(self) -> None:
        self.lstm = LSTM(shape=(5, 10), architecture=[5, 5, 5], dropout_rate=0.1)
        self.conv_lstm = LSTM(shape=(5, 10), architecture=[5, 5, 5], dropout_rate=0.1, convolutional_layer=True,
                              kernel_size=1, padding='same', filters=64, pool_size=2)
        self.bi_lstm = LSTM(shape=(5, 10), architecture=[5, 5, 5], dropout_rate=0.1, bidirectional=True,
                            merge_mode='concat')
        self.conv_bi_lstm = LSTM(shape=(5, 10), architecture=[5, 5, 5], dropout_rate=0.1, bidirectional=True,
                                 merge_mode='concat', convolutional_layer=True, kernel_size=1, padding='same',
                                 filters=64, pool_size=2)

    def test_compile_model(self):
        self.lstm.compile_model(loss=tf.losses.mean_squared_error, metrics=[tf.metrics.mean_absolute_error],
                                optimizer=tf.optimizers.Adam())
        model = self.lstm.get_model()
        self.assertIsInstance(model.compiled_loss._loss_metric, tf.keras.metrics.Mean)
        self.assertTrue(model._is_compiled)
        self.assertIsInstance(model.optimizer, tf.keras.optimizers.Adam)

    def test_get_model(self):
        model = self.lstm.get_model()
        self.assertIsInstance(model, keras.Model)
        self.assertTrue(model.built)
        self.assertEqual(len(model.layers), 8)
        self.assertIsInstance(model.layers[0], keras.layers.InputLayer)
        for i in range(1, 6, 2):
            self.assertIsInstance(model.layers[i], keras.layers.Dropout)
            self.assertIsInstance(model.layers[i + 1], keras.layers.LSTM)
        self.assertIsInstance(model.layers[-1], keras.layers.Dense)

    def test_get_model_conv_lstm(self):
        model = self.conv_lstm.get_model()
        self.assertIsInstance(model, keras.Model)
        self.assertTrue(model.built)
        self.assertEqual(len(model.layers), 11)
        self.assertIsInstance(model.layers[0], keras.layers.InputLayer)
        self.assertIsInstance(model.layers[1], keras.layers.Conv1D)
        self.assertIsInstance(model.layers[2], keras.layers.MaxPool1D)
        self.assertIsInstance(model.layers[3], keras.layers.TimeDistributed)
        for i in range(4, 9, 2):
            self.assertIsInstance(model.layers[i], keras.layers.Dropout)
            self.assertIsInstance(model.layers[i + 1], keras.layers.LSTM)
        self.assertIsInstance(model.layers[-1], keras.layers.Dense)

    def test_get_model_bi_lstm(self):
        model = self.bi_lstm.get_model()
        self.assertIsInstance(model, keras.Model)
        self.assertTrue(model.built)
        self.assertEqual(len(model.layers), 8)
        self.assertIsInstance(model.layers[0], keras.layers.InputLayer)
        for i in range(1, 5, 2):
            self.assertIsInstance(model.layers[i], keras.layers.Dropout)
            self.assertIsInstance(model.layers[i + 1], keras.layers.Bidirectional)
        self.assertIsInstance(model.layers[-3], keras.layers.Dropout)
        self.assertIsInstance(model.layers[-2], keras.layers.LSTM)
        self.assertIsInstance(model.layers[-1], keras.layers.Dense)

    def test_get_model_conv_bi_lstm(self):
        model = self.conv_bi_lstm.get_model()
        self.assertIsInstance(model, keras.Model)
        self.assertTrue(model.built)
        self.assertEqual(len(model.layers), 11)
        self.assertIsInstance(model.layers[0], keras.layers.InputLayer)
        self.assertIsInstance(model.layers[1], keras.layers.Conv1D)
        self.assertIsInstance(model.layers[2], keras.layers.MaxPool1D)
        self.assertIsInstance(model.layers[3], keras.layers.TimeDistributed)
        for i in range(4, 7, 2):
            self.assertIsInstance(model.layers[i], keras.layers.Dropout)
            self.assertIsInstance(model.layers[i + 1], keras.layers.Bidirectional)
        self.assertIsInstance(model.layers[-3], keras.layers.Dropout)
        self.assertIsInstance(model.layers[-2], keras.layers.LSTM)
        self.assertIsInstance(model.layers[-1], keras.layers.Dense)

    @patch('builtins.print')
    def test_get_summary(self, mock_stdout):
        self.lstm.get_summary()
        mock_stdout.assert_called()
