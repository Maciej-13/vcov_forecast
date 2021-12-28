import pickle
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, Union, List, Optional


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
    history: Dict[int, Dict[int, Trade]] = field(default_factory=dict)

    def __register_single_asset(self, stamp: int, asset: str, quantity: int, price: float, buy: bool,
                                fee_multiplier: Optional[float]) -> None:
        if stamp not in self.history:
            self.history[stamp] = {
                0: Trade(asset=asset, quantity=quantity, price=price, buy=buy, fee_multiplier=fee_multiplier)}
        else:
            k: int = list(self.history[stamp])[-1] + 1
            self.history[stamp].update(
                {k: Trade(asset=asset, quantity=quantity, price=price, buy=buy, fee_multiplier=fee_multiplier)})

    def __register_multiple_assets(self, stamp: int, asset: List[str], quantity: List[int], price: np.ndarray,
                                   buy: bool, fee_multiplier: Optional[List[Optional[float]]]) -> None:
        if fee_multiplier is None:
            fee_multiplier = [None] * len(asset)
        if not len(asset) == len(quantity) == len(price) == len(fee_multiplier):
            raise ValueError("Length of arguments: asset, quantity, price, fee multiplier must be the same!")

        if stamp not in self.history:
            self.history[stamp] = {i: Trade(asset=asset[i], quantity=quantity[i], price=price[i], buy=buy,
                                            fee_multiplier=fee_multiplier[i]) for i in range(len(asset))}
        else:
            k: int = list(self.history[stamp])[-1] + 1
            self.history[stamp].update(
                {k + i: Trade(asset=asset[i], quantity=quantity[i], price=price[i], buy=buy,
                              fee_multiplier=fee_multiplier[i]) for i in range(len(asset))}
            )

    def register(self, stamp: int, asset: Union[str, List[str]], quantity: Union[int, List[int]],
                 price: Union[float, np.ndarray], buy: bool,
                 fee_multiplier: Union[Optional[float], List[Optional[float]]]) -> None:
        if isinstance(asset, list):
            self.__register_multiple_assets(
                stamp=stamp,  # type: ignore
                asset=asset,  # type: ignore
                quantity=quantity,  # type: ignore
                price=price,  # type: ignore
                buy=buy,
                fee_multiplier=fee_multiplier  # type: ignore
            )
        else:
            self.__register_single_asset(
                stamp=stamp,  # type: ignore
                asset=asset,  # type: ignore
                quantity=quantity,  # type: ignore
                price=price,  # type: ignore
                buy=buy,
                fee_multiplier=fee_multiplier  # type: ignore
            )

    def save(self, path_filename: str) -> None:
        with open(f'{path_filename}.pickle', 'wb') as f:
            pickle.dump(self.history, f)
