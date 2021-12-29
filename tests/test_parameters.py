import pytest

from vcov.modules.portfolio.parameters import StrategyParameters, VALID_COV, RETURNS, OPTIMIZE


def test_parameters():
    params = StrategyParameters(100, 'sample_cov', portfolio_value=1000)
    assert isinstance(params, StrategyParameters)
    assert params.window == 100
    assert params.covariance_model == 'sample_cov'
    assert params.rebalancing is None
    assert params.fee_multiplier is None


def test_parameters_rebalancing():
    params = StrategyParameters(100, 'sample_cov', rebalancing=100, portfolio_value=1000)
    assert params.rebalancing == 100
    assert params.fee_multiplier is None


def test_parameters_fees():
    params = StrategyParameters(100, 'sample_cov', fee_multiplier=0.01, portfolio_value=1000)
    assert params.fee_multiplier == 0.01
    assert params.rebalancing is None


def test_parameters_validate_covariance():
    with pytest.raises(ValueError) as e:
        StrategyParameters(100, 'test', 1000)

    assert e.value.args[0] == f"Unknown parameter: test! Use one of: {VALID_COV}"


def test_parameters_classical():
    params = StrategyParameters(100, 'sample_cov', rebalancing=100, portfolio_value=1000)
    assert params.classical
    params = StrategyParameters(100, 'lstm', rebalancing=100, portfolio_value=1000)
    assert not params.classical


def test_parameters_returns():
    params = StrategyParameters(100, 'sample_cov', rebalancing=100, portfolio_value=1000)
    assert params.returns == 'mean_historical_return'
    params = StrategyParameters(100, 'sample_cov', returns='capm_return', rebalancing=100, portfolio_value=1000)
    assert params.returns == 'capm_return'


def test_parameters_returns_error():
    with pytest.raises(ValueError) as e:
        StrategyParameters(100, 'sample_cov', returns='test', rebalancing=100, portfolio_value=1000)

    assert e.value.args[0] == f"Unknown parameter: test! Use one of: {RETURNS}"


def test_parameters_optimize():
    params = StrategyParameters(100, 'sample_cov', rebalancing=100, portfolio_value=1000)
    assert params.optimize == 'min_volatility'
    params = StrategyParameters(100, 'sample_cov', optimize='max_sharpe', rebalancing=100, portfolio_value=1000)
    assert params.optimize == 'max_sharpe'


def test_parameters_optimize_error():
    with pytest.raises(ValueError) as e:
        StrategyParameters(100, 'sample_cov', optimize='test', rebalancing=100, portfolio_value=1000)

    assert e.value.args[0] == f"Unknown parameter: test! Use one of: {OPTIMIZE}"
