from dataclasses import dataclass, field
from typing import Dict, List, Union


@dataclass
class Portfolio:
    assets: List[str]
    weights: Dict[str, float] = field(default_factory=dict)
    stocks: Dict[str, int] = field(default_factory=dict)

    def __post_init__(self):
        if self.weights:
            self._weights_validation()

    def update_weights(self, weights: Dict[str, float]) -> None:
        self.weights = weights
        self._weights_validation()

    def update_stocks(self, stocks: Dict[str, int]) -> None:
        self.stocks.clear()
        self.stocks.update(stocks)

    def remove_asset(self, assets: Union[str, List[str]]) -> None:
        if isinstance(assets, str):
            self.assets.remove(assets)
            self.weights.pop(assets, None)
            self.stocks.pop(assets, None)
        else:
            self.assets = [a for a in self.assets if a not in assets]
            self.weights = {a: w for a, w in self.weights.items() if a not in assets}
            self.stocks = {a: s for a, s in self.stocks.items() if a not in assets}

    def add_asset(self, assets: Union[str, List[str]], weights: Dict[str, float], stocks: Dict[str, int]) -> None:
        self.update_weights(weights)
        self.update_stocks(stocks)
        if isinstance(assets, str):
            self.assets.append(assets)
        else:
            self.assets.extend(assets)

    def _weights_validation(self):
        if round(sum(self.weights.values()), 2) != 1:
            raise ValueError("The weights do not add up to 1!")
