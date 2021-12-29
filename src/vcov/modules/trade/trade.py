import pickle
from dataclasses import dataclass, field
from typing import Dict, Union, List, Optional
from pandas import Series, Timestamp


@dataclass
class Trade:
    asset: str
    quantity: Union[int, float]
    price: float
    buy: bool
    fee_multiplier: Optional[float]

    def __str__(self) -> str:
        return f"asset: {self.asset}\n" \
               f"quantity: {self.quantity}\n" \
               f"price: {self.price}\n" \
               f"position: {'buy' if self.buy else 'sell'}\n" \
               f"fee: {self.fee_multiplier}"

    def calculate_fees(self) -> Union[float, int]:
        if self.fee_multiplier is not None:
            return self.quantity * self.price * self.fee_multiplier
        return 0


@dataclass
class TradeHistory:
    history: Dict[Timestamp, Dict[int, Trade]] = field(default_factory=dict)

    def __register_single_asset(self, stamp: Timestamp, asset: str, quantity: int, price: float, buy: bool,
                                fee_multiplier: Optional[float]) -> None:
        if quantity == 0:
            pass

        if stamp not in self.history:
            self.history[stamp] = {
                0: Trade(asset=asset, quantity=quantity, price=price, buy=buy, fee_multiplier=fee_multiplier)}
        else:
            k: int = list(self.history[stamp])[-1] + 1
            self.history[stamp].update(
                {k: Trade(asset=asset, quantity=quantity, price=price, buy=buy, fee_multiplier=fee_multiplier)})

    def __register_multiple_assets(self, stamp: Timestamp, asset: List[str], quantity: List[int], price: Series,
                                   buy: bool, fee_multiplier: Optional[float]) -> None:
        if not len(asset) == len(quantity) == len(price):
            raise ValueError("Length of arguments: asset, quantity, price must be the same!")

        if stamp not in self.history:
            for i in range(len(asset)):
                self.history[stamp] = {i: Trade(asset=asset[i], quantity=quantity[i], price=price[i], buy=buy,
                                                fee_multiplier=fee_multiplier) for i in range(len(asset)) if
                                       quantity[i] != 0}
        else:
            k: int = list(self.history[stamp])[-1] + 1
            self.history[stamp].update(
                {k + i: Trade(asset=asset[i], quantity=quantity[i], price=price[i], buy=buy,
                              fee_multiplier=fee_multiplier) for i in range(len(asset)) if quantity[i] != 0}
            )

    def register(self, stamp: Timestamp, asset: Union[str, List[str]], quantity: Union[int, List[int]],
                 price: Union[float, Series], buy: bool,
                 fee_multiplier: Optional[float]) -> None:
        if isinstance(asset, list):
            self.__register_multiple_assets(
                stamp=stamp,
                asset=asset,  # type: ignore
                quantity=quantity,  # type: ignore
                price=price,  # type: ignore
                buy=buy,
                fee_multiplier=fee_multiplier
            )
        else:
            self.__register_single_asset(
                stamp=stamp,
                asset=asset,  # type: ignore
                quantity=quantity,  # type: ignore
                price=price,  # type: ignore
                buy=buy,
                fee_multiplier=fee_multiplier
            )

    def save(self, path_filename: str) -> None:
        with open(f'{path_filename}.pickle', 'wb') as f:
            pickle.dump(self.history, f)
