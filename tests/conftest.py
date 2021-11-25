import pytest
import os

from vcov.modules.data_handling.input_handler import InputHandler
from vcov.modules.data_handling.covariance_handler import CovarianceHandler
from vcov.modules.models.lstm_model import KerasTCN, LSTM


@pytest.fixture()
def data_dir():
    wd = os.path.dirname(os.path.realpath(__file__))
    print(wd)
    if "tests" in wd:
        return wd + "/data/"
    else:
        return wd + "/tests/data/"


@pytest.fixture()
def data_path(data_dir):
    return data_dir + "/AAPL_F.csv"


@pytest.fixture()
def prices(data_path):
    return InputHandler(data_path, assets=['AAPL', 'F'], returns=False).get_data()


@pytest.fixture()
def returns(data_dir):
    return InputHandler(data_dir + '/data_short.csv', assets=['AAPL', 'BAC', 'MSFT', 'GOOG'], returns=True).get_data()


@pytest.fixture()
def single_prices(data_dir):
    return InputHandler(data_dir + '/data_short.csv',
                        assets=['AAPL', 'BAC', 'MSFT', 'GOOG'], returns=False).get_data()["AAPL"]


@pytest.fixture()
def returns_no_idx(returns):
    data = returns.reset_index(drop=True)
    return data


@pytest.fixture()
def cov_data(data_dir):
    data = InputHandler(data_dir + '/data_short.csv', assets=['AAPL', 'BAC', 'MSFT', 'GOOG'], column='Close',
                        returns=True).get_data()
    cov = CovarianceHandler(lookback=15, n_assets=4)
    return cov.split_covariance_to_wide(cov.calculate_rolling_covariance_matrix(data))


@pytest.fixture()
def tcn():
    return KerasTCN(shape=(5, 5), nb_filters=25, kernel_size=3, nb_stacks=1, dilations=(1, 2), dropout=0.2,
                    padding="same")


@pytest.fixture()
def lstm():
    return LSTM(shape=(5, 10), architecture=[5, 5, 5], dropout_rate=0.1)
