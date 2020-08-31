from typing import Dict, Iterable, Optional, Union
from torch import Tensor

Gamma = Optional[Tensor]
Gradient = Optional[Tensor]
Parameter = Iterable[Union[Tensor, Dict]]


def diminish1(n: int) -> float:
    return .5 ** n


def diminish2(n: int) -> float:
    return 1 / n


lr_fn_dict = dict(
    C1=lambda n: 1e-1,
    C2=lambda n: 1e-2,
    C3=lambda n: 1e-3,
    D1=diminish1,
    D2=diminish2,
)
