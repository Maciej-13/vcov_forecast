from tensorflow import keras
from tensorflow.keras import layers
from tcn import TCN, tcn_full_summary
from typing import Optional, Tuple, List, Callable, Dict, Any, Union
from keras import Model


class LSTM:

    def __init__(self, shape: Tuple[int, ...], architecture: List[int], dropout_rate: float,
                 batch_size: Optional[int] = None, bidirectional: bool = False, merge_mode: Optional[str] = None,
                 convolutional_layer: bool = False, kernel_size: Optional[int] = None, padding: Optional[str] = None,
                 filters: Optional[int] = None, pool_size: Optional[int] = None) -> None:
        self._shape = shape
        self._architecture = architecture
        self._batch_size = batch_size
        self._dropout = dropout_rate
        self._bi = bidirectional
        self._merge = merge_mode if self._bi else None
        self._conv = convolutional_layer
        self._kernel_size = kernel_size if self._conv else None
        self._padding = padding if self._conv else None
        self._filters = filters if self._conv else None
        self._pool_size = pool_size if self._conv else None
        self._model: Model = self._create_model()

    def compile_model(self, loss: Callable[[Any], Any], optimizer: Callable[[Any], Any],
                      metrics: List[Callable[[Any], Any]], **kwargs) -> None:
        self._model.compile(optimizer=optimizer, loss=loss, metrics=metrics, **kwargs)

    def get_model(self) -> Model:
        return self._model

    def get_summary(self) -> Any:
        return self._model.summary()

    @staticmethod
    def get_attributes() -> Dict[str, Any]:
        return dict((x, y) for x, y in LSTM.__dict__.items() if x[2:] != 'model')

    def _create_model(self) -> Model:
        inputs = layers.Input(shape=self._shape)

        if self._conv:
            model = layers.Conv1D(filters=self._filters, kernel_size=self._kernel_size, padding=self._padding,
                                  activation='relu')(inputs)
            model = layers.MaxPool1D(pool_size=self._pool_size)(model)
            model = layers.TimeDistributed(layers.Flatten())(model)
        else:
            model = inputs

        for u in self._architecture[:-1]:
            model = layers.Dropout(self._dropout)(model)
            model = self._recurrent_layer(u, return_sequences=True)(model)

        model = layers.Dropout(self._dropout)(model)
        model = layers.LSTM(self._architecture[-1], return_sequences=False)(model)
        outputs = layers.Dense(self._shape[-1])(model)
        final_model = keras.Model(inputs, outputs)

        return final_model

    def _recurrent_layer(self, units: int, **kwargs) -> Union[layers.LSTM, layers.Bidirectional]:
        if self._bi:
            return layers.Bidirectional(layers.LSTM(units, **kwargs), merge_mode=self._merge)
        else:
            return layers.LSTM(units, **kwargs)


class KerasTCN:

    def __init__(self, shape: Tuple[int, ...], nb_filters: int, kernel_size: int, nb_stacks: int,
                 dilations: Tuple[int, ...], padding: str, dropout: float, **kwargs) -> None:
        self._model = self._create_model(shape=shape, nb_filters=nb_filters, kernel_size=kernel_size,
                                         nb_stacks=nb_stacks, dilations=dilations, padding=padding,
                                         dropout=dropout, **kwargs)

    def compile_model(self, loss: Callable[[Any], Any], optimizer: Callable[[Any], Any],
                      metrics: List[Callable[[Any], Any]], **kwargs) -> None:
        self._model.compile(optimizer=optimizer, loss=loss, metrics=metrics, **kwargs)

    def get_model(self) -> keras.Sequential:
        return self._model

    def get_summary(self, expand_residual_blocks: bool = True) -> Any:
        return tcn_full_summary(self._model, expand_residual_blocks=expand_residual_blocks)

    @staticmethod
    def _create_model(shape: Tuple[int, ...], nb_filters: int, kernel_size: int, nb_stacks: int,
                      dilations: Tuple[int, ...], padding: str, dropout: float, **kwargs) -> keras.Sequential:
        tcn = TCN(input_shape=shape, nb_filters=nb_filters, kernel_size=kernel_size, nb_stacks=nb_stacks,
                  dilations=dilations, padding=padding, dropout_rate=dropout, **kwargs)
        outputs = layers.Dense(shape[-1])
        final_model = keras.Sequential([tcn, outputs])
        return final_model
