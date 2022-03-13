from typing import Tuple, List, Optional, Dict, Union
from dataclasses import dataclass

ParamsDict = Dict[str, Union[str, int, float, Tuple[int, ...]]]


@dataclass
class LstmHyperparameters:
    epochs: int = 50
    batch_size: int = 2
    length: int = 4
    architecture: Tuple[int, ...] = (20, 10)
    dropout_rate: float = 0.1
    stopping_patience: int = 5

    # Bidirectional LSTM params
    bidirectional: bool = False
    merge_mode: Optional[str] = None

    # Conv-Lstm Params
    convolutional_layer: bool = False
    kernel_size: Optional[int] = None
    padding: Optional[str] = None
    filters: Optional[int] = None
    pool_size: Optional[int] = None

    def to_dict(self) -> ParamsDict:
        return self.__dict__


@dataclass
class TCNHyperparameters:
    nb_filters: int
    kernel_size: int
    nb_stacks: int
    dilations: Tuple[int, ...]
    padding: str
    dropout: float = 0.1

    def to_dict(self) -> ParamsDict:
        return self.__dict__


@dataclass
class GluonHyperparameters:
    cell_type: str
    batch_size: int
    num_cells: int
    num_layers: int

    epochs: int = 100
    dropout_rate: float = 0.1
    patience: int = 5
    num_batches_per_epoch: int = 5

    learning_rate: float = 0.1
    learning_rate_fullrank: float = 0.01
    minimum_learning_rate: float = 0.001

    rank: Optional[int] = None
    num_eval_samples: Optional[int] = None
    conditioning_length: int = 200
    hybridize: bool = False
    target_dim_sample: Optional[int] = None
    lags_seq: Optional[List[int]] = None
    scaling: Optional[bool] = False
    context_length: Optional[int] = None
    low_rank: bool = False

    def to_dict(self) -> ParamsDict:
        return self.__dict__
