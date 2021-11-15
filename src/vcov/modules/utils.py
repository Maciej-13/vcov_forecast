from typing import Optional, Union, List
from vcov.modules.data_handling.input_handler import InputHandler
from vcov.modules.data_handling.covariance_handler import CovarianceHandler
from vcov.modules.data_handling.dataset import KerasDataset


def get_prepared_generator(path: str, assets: List[str], lookback: int, length: int,
                           forward_shift: Optional[int] = None, val_size: Union[float, int] = 0.2, **kwargs):
    if 0 < val_size < 1:
        train, val = InputHandler(path, assets).train_test_split(val_size)
        cov_handler = CovarianceHandler(lookback, n_assets=len(assets))
        train_cholesky = cov_handler.cholesky_transformation(cov_handler.calculate_rolling_covariance_matrix(train))
        val_cholesky = cov_handler.cholesky_transformation(cov_handler.calculate_rolling_covariance_matrix(val))
        train_gen = KerasDataset(train_cholesky, length=length, forward_shift=forward_shift, **kwargs).get_generator()
        val_gen = KerasDataset(val_cholesky, length=length, forward_shift=forward_shift, **kwargs).get_generator()
        return train_gen, val_gen

    elif val_size == 0:
        data = InputHandler(path, assets).get_data()
        cov_handler = CovarianceHandler(lookback, n_assets=len(assets))
        cholesky_data = cov_handler.cholesky_transformation(cov_handler.calculate_rolling_covariance_matrix(data))
        gen = KerasDataset(cholesky_data, length=length, forward_shift=forward_shift, **kwargs).get_generator()
        return gen

    else:
        raise ValueError("Parameter val_size must be greater of equal to zero and not higher than 1!")
