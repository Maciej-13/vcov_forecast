import pytest

from vcov.modules.portfolio.parameters import StrategyParameters, VALID_COV


def test_parameters():
    params = StrategyParameters(100, 'sample_cov', portfolio_value=1000)
    assert isinstance(params, StrategyParameters)
    assert params.window == 100
    assert params.covariance_model == 'sample_cov'
    assert params.rebalancing is None
    assert params.fees is None


def test_parameters_rebalancing():
    params = StrategyParameters(100, 'sample_cov', rebalancing=100, portfolio_value=1000)
    assert params.rebalancing == 100
    assert params.fees is None


def test_parameters_fees():
    params = StrategyParameters(100, 'sample_cov', fees=0.01, portfolio_value=1000)
    assert params.fees == 0.01
    assert params.rebalancing is None


def test_parameters_validate_covariance():
    with pytest.raises(ValueError) as e:
        StrategyParameters(100, 'test', 1000)

    assert e.value.args[0] == f"Unknown covariance estimation method: test! Use one of: {VALID_COV}"
