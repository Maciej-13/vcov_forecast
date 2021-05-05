import unittest
from unittest.mock import patch

import tensorflow as tf
from tensorflow import keras

from modules.models.keras_model import LSTM


class TestLSTM(unittest.TestCase):

    def setUp(self) -> None:
        self.lstm = LSTM(shape=(1, 10), architecture=[10, 10, 10], batch_size=1, dropout_rate=0.1)

    def test_compile_model(self):
        pass

    def test_train(self):
        pass

    def test_evaluate(self):
        pass

    def test_predict(self):
        pass

    def test_save(self):
        pass

    def test_multistep_ahead_forecast(self):
        pass

    def test_get_model(self):
        pass

    @patch('builtins.print')
    def test_get_summary(self, mock_stdout):
        self.lstm.get_summary()
        mock_stdout.assert_called()
