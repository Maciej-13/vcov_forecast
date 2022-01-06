import pytest

from vcov.modules.portfolio.portfolio import Portfolio


def test_portfolio():
    a = ['a', 'b']
    w = {'a': 0.5, 'b': 0.5}
    port = Portfolio(assets=a, weights=w)
    assert isinstance(port, Portfolio)
    assert port.assets == a
    assert port.weights == w


def test_portfolio_error():
    with pytest.raises(ValueError) as e:
        Portfolio(assets=['a', 'b'], weights={'a': 0.1, 'b': 0.1})

    assert e.value.args[0] == "The weights do not add up to 1!"


def test_update_weights():
    w = {'a': 0.5, 'b': 0.5}
    port = Portfolio(assets=['a', 'b'], weights=w)
    assert port.weights == w
    new_w = {'a': 0.9, 'b': 0.1}
    port.update_weights(new_w)
    assert port.weights == new_w


def test_update_weights_error():
    w = {'a': 0.5, 'b': 0.5}
    port = Portfolio(assets=['a', 'b'], weights=w)
    assert port.weights == w
    new_w = {'a': 0.9, 'b': 0.05}
    with pytest.raises(ValueError) as e:
        port.update_weights(new_w)

    assert e.value.args[0] == "The weights do not add up to 1!"


def test_remove_asset_single():
    port = Portfolio(assets=['a', 'b'], weights={'a': 0.5, 'b': 0.5})
    assert len(port.assets) == 2
    port.remove_asset('a')
    assert len(port.assets) == 1
    assert port.assets == ['b']
    assert port.weights == {'b': 0.5}


def test_remove_asset_multiple():
    port = Portfolio(assets=['a', 'b', 'c'], weights={'a': 0.3, 'b': 0.3, 'c': 0.4}, stocks={'a': 3, 'b': 1, 'c': 1})
    assert len(port.assets) == 3
    port.remove_asset(['a', 'b'])
    assert len(port.assets) == 1
    assert port.assets == ['c']
    assert port.weights == {'c': 0.4}
    assert port.stocks == {'c': 1}


def test_add_asset_single():
    port = Portfolio(assets=['a', 'b'], weights={'a': 0.5, 'b': 0.5})
    assert len(port.assets) == 2
    port.add_asset('c', {'c': 0.2, 'a': 0.2, 'b': 0.6}, {'a': 2, 'b': 5, 'c': 1})
    assert port.assets == ['a', 'b', 'c']
    assert port.weights == {'a': 0.2, 'b': 0.6, 'c': 0.2}
    assert port.stocks == {'a': 2, 'b': 5, 'c': 1}


def test_add_assets_multiple():
    port = Portfolio(assets=['a', 'b'], weights={'a': 0.5, 'b': 0.5})
    assert len(port.assets) == 2
    port.add_asset(['c', 'd'], {'a': 0.1, 'b': 0.1, 'c': 0.3, 'd': 0.5}, stocks={'a': 4, 'b': 1, 'c': 3, 'd': 20})
    assert len(port.assets) == 4
    assert port.weights == {'a': 0.1, 'b': 0.1, 'c': 0.3, 'd': 0.5}
    assert port.stocks == {'a': 4, 'b': 1, 'c': 3, 'd': 20}


def test_update_stocks():
    port = Portfolio(assets=['a', 'b'], weights={'a': 0.5, 'b': 0.5}, stocks={'a': 2, 'b': 5})
    assert len(port.stocks) == 2
    port.update_stocks({'a': 6})
    assert 'b' not in port.stocks
    assert port.stocks == {'a': 6}
