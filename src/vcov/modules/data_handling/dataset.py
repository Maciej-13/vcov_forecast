import numpy as np
import pandas as pd

from typing import Union, Optional, Tuple
from keras.preprocessing.sequence import TimeseriesGenerator
from pandas.core.frame import DataFrame
from pandas.core.series import Series


class KerasDataset:

    def __init__(self, data: Union[DataFrame, Series], length: int, forward_shift: Optional[int] = None, **kwargs):
        self.data = data
        self.shift = forward_shift if forward_shift is None else forward_shift - 1 if forward_shift > 1 else 0
        self.length = length
        self._generator = self._create_generator(**kwargs)

    def get_generator(self) -> TimeseriesGenerator:
        return self._generator

    def _create_generator(self, **kwargs) -> TimeseriesGenerator:
        data: np.ndarray = self._prepare_arrays()
        if self.shift is None:
            return TimeseriesGenerator(data, data, length=self.length, **kwargs)
        data, targets = self._shift_array(data)
        return TimeseriesGenerator(data, targets, length=self.length, **kwargs)

    def _prepare_arrays(self) -> np.ndarray:
        if isinstance(self.data, pd.Series):
            return np.array(self.data).astype(np.float32).reshape((len(self.data), 1))

        else:
            arrays = tuple(np.array(self.data[i]).reshape((len(self.data), 1)) for i in self.data.columns)
            matrix = np.hstack(arrays).astype(np.float32)
            return matrix

    def _shift_array(self, array: np.ndarray) -> Union[Tuple[np.ndarray, np.ndarray], np.ndarray]:
        if self.shift is not None:
            target = array[self.shift:]
            values = array[:(len(array) - self.shift)]
            return values.astype(np.float32), target.astype(np.float32)
        else:
            return array
