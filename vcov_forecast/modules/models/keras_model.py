from tensorflow import keras
from tensorflow.keras import layers
from tcn import TCN, tcn_full_summary

from beartype import beartype
from beartype.cave import NoneType


class LSTM:

    @beartype
    def __init__(self, shape: tuple, architecture: list, dropout_rate: float, batch_size: (int, NoneType) = None,
                 bidirectional: bool = False, merge_mode: (str, NoneType) = None, convolutional_layer: bool = False,
                 kernel_size: (int, NoneType) = None, padding: (str, NoneType) = None, filters: (int, NoneType) = None,
                 pool_size: (int, NoneType) = None):
        self.__shape = shape
        self.__architecture = architecture
        self.__batch_size = batch_size
        self.__dropout = dropout_rate
        self.__bi = bidirectional
        self.__merge = merge_mode if self.__bi else None
        self.__conv = convolutional_layer
        self.__kernel_size = kernel_size if self.__conv else None
        self.__padding = padding if self.__conv else None
        self.__filters = filters if self.__conv else None
        self.__pool_size = pool_size if self.__conv else None
        self.__model = self.__create_model()

    @beartype
    def compile_model(self, loss, optimizer, metrics: list, **kwargs):
        self.__model.compile(optimizer=optimizer, loss=loss, metrics=metrics, **kwargs)

    def get_model(self):
        return self.__model

    def get_summary(self):
        return self.__model.summary()

    def __create_model(self):
        inputs = layers.Input(shape=self.__shape)

        if self.__conv:
            model = layers.Conv1D(filters=self.__filters, kernel_size=self.__kernel_size, padding=self.__padding,
                                  activation='relu')(inputs)
            model = layers.MaxPool1D(pool_size=self.__pool_size)(model)
            model = layers.TimeDistributed(layers.Flatten())(model)
        else:
            model = inputs

        for u in self.__architecture[:-1]:
            model = layers.Dropout(self.__dropout)(model)
            model = self.__recurrent_layer(u, return_sequences=True)(model)

        model = layers.Dropout(self.__dropout)(model)
        model = layers.LSTM(self.__architecture[-1], return_sequences=False)(model)
        outputs = layers.Dense(self.__shape[-1])(model)
        final_model = keras.Model(inputs, outputs)

        return final_model

    @beartype
    def __recurrent_layer(self, units: int, **kwargs):
        if self.__bi:
            return layers.Bidirectional(layers.LSTM(units, **kwargs), merge_mode=self.__merge)
        else:
            return layers.LSTM(units, **kwargs)
