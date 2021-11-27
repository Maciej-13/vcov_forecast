import pytest

from vcov.modules.strategy.portfolio import Portfolio


def test_portfolio():
    a = ['a', 'b']
    w = {'a': 0.5, 'b': 0.5}
    p = {'a': 1, 'b': 1}
    port = Portfolio(assets=a, weights=w, buy_prices=p)
    assert isinstance(port, Portfolio)
    assert port.assets == a
    assert port.weights == w
    assert port.buy_prices == p


def test_portfolio_error():
    with pytest.raises(ValueError) as e:
        Portfolio(assets=['a', 'b'], weights={'a': 0.1, 'b': 0.1}, buy_prices={'a': 1, 'b': 1})

    assert e.value.args[0] == "The weights do not add up to 1!"


def test_update_weights():
    w = {'a': 0.5, 'b': 0.5}
    port = Portfolio(assets=['a', 'b'], weights=w, buy_prices={'a': 1, 'b': 1})
    assert port.weights == w
    new_w = {'a': 0.9, 'b': 0.1}
    port.update_weights(new_w)
    assert port.weights == new_w


def test_update_weights_error():
    w = {'a': 0.5, 'b': 0.5}
    port = Portfolio(assets=['a', 'b'], weights=w, buy_prices={'a': 1, 'b': 1})
    assert port.weights == w
    new_w = {'a': 0.9, 'b': 0.05}
    with pytest.raises(ValueError) as e:
        port.update_weights(new_w)

    assert e.value.args[0] == "The weights do not add up to 1!"


def test_update_buy_prices():
    p = {'a': 1, 'b': 1}
    port = Portfolio(assets=['a', 'b'], weights={'a': 0.5, 'b': 0.5}, buy_prices=p)
    assert port.buy_prices == p
    new_p = {'a': 2, 'b': 0.5}
    port.update_buy_prices(new_p)
    assert port.buy_prices == new_p


def test_buy_prices_error():
    p = {'a': 1, 'b': 1}
    port = Portfolio(assets=['a', 'b'], weights={'a': 0.5, 'b': 0.5}, buy_prices=p)
    assert port.buy_prices == p
    new_p = {'a': 2, 'b': -0.5}
    with pytest.raises(ValueError) as e:
        port.update_buy_prices(new_p)

    assert e.value.args[0] == "Prices cannot be negative!"


def test_remove_asset_single():
    port = Portfolio(assets=['a', 'b'], weights={'a': 0.5, 'b': 0.5}, buy_prices={'a': 1, 'b': 1})
    assert len(port.assets) == 2
    port.remove_asset('a')
    assert len(port.assets) == 1
    assert port.assets == ['b']
    assert port.weights == {'b': 0.5}
    assert port.buy_prices == {'b': 1}


def test_remove_asset_multiple():
    port = Portfolio(assets=['a', 'b', 'c'], weights={'a': 0.3, 'b': 0.3, 'c': 0.4},
                     buy_prices={'a': 1, 'b': 1, 'c': 2})
    assert len(port.assets) == 3
    port.remove_asset(['a', 'b'])
    assert len(port.assets) == 1
    assert port.assets == ['c']
    assert port.weights == {'c': 0.4}
    assert port.buy_prices == {'c': 2}


def test_add_asset_single():
    port = Portfolio(assets=['a', 'b'], weights={'a': 0.5, 'b': 0.5}, buy_prices={'a': 1, 'b': 1})
    assert len(port.assets) == 2
    port.add_asset('c', {'c': 0.2, 'a': 0.2, 'b': 0.6}, buy_prices=0.35)
    assert port.assets == ['a', 'b', 'c']
    assert port.weights == {'a': 0.2, 'b': 0.6, 'c': 0.2}
    assert port.buy_prices['c'] == 0.35


def test_add_assets_multiple():
    port = Portfolio(assets=['a', 'b'], weights={'a': 0.5, 'b': 0.5}, buy_prices={'a': 1, 'b': 1})
    assert len(port.assets) == 2
    port.add_asset(['c', 'd'], {'a': 0.1, 'b': 0.1, 'c': 0.3, 'd': 0.5}, buy_prices={'c': 2, 'd': 0.1})
    assert len(port.assets) == 4
    assert port.weights == {'a': 0.1, 'b': 0.1, 'c': 0.3, 'd': 0.5}
    assert port.buy_prices['c'] == 2
    assert port.buy_prices['d'] == 0.1
