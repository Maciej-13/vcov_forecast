import pytest

from vcov.modules.strategy.parameters import StrategyParameters, VALID_COV


def test_parameters():
    params = StrategyParameters(100, 'sample_cov')
    assert isinstance(params, StrategyParameters)
    assert params.window == 100
    assert params.covariance == 'sample_cov'
    assert params.rebalancing is None
    assert params.fees is None
    assert not params.additional
    assert len(params.additional.keys()) == 0


def test_parameters_rebalancing():
    params = StrategyParameters(100, 'sample_cov', rebalancing=100)
    assert params.rebalancing == 100
    assert params.fees is None
    assert not params.additional
    assert len(params.additional.keys()) == 0


def test_parameters_fees():
    params = StrategyParameters(100, 'sample_cov', fees=0.01)
    assert params.fees == 0.01
    assert params.rebalancing is None
    assert not params.additional
    assert len(params.additional.keys()) == 0


def test_parameters_additional():
    params = StrategyParameters(100, 'sample_cov', rebalancing=20, additional={'test': 1, 'test_1': 'abc'})
    assert params.window == 100
    assert params.covariance == 'sample_cov'
    assert params.rebalancing == 20
    assert isinstance(params.additional, dict)
    assert len(params.additional.keys()) == 2
    assert params.additional['test'] == 1
    assert params.additional['test_1'] == 'abc'


def test_parameters_validate_covariance():
    with pytest.raises(ValueError) as e:
        StrategyParameters(100, 'test')

    assert e.value.args[0] == f"Unknown covariance estimation method: test! Use one of: {VALID_COV}"
