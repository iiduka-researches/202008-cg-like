from math import sqrt
from typing import Dict, Iterable, Optional, Union
from torch import Tensor

Gamma = Optional[Tensor]
Gradient = Optional[Tensor]
Parameter = Iterable[Union[Tensor, Dict]]


def diminish0(n: int) -> float:
    return 1 / sqrt(n)


def diminish1(n: int) -> float:
    return .5 ** n


def diminish2(n: int) -> float:
    return 1 / n


lr_fn_dict = dict(
    C1=lambda n: 1e-1,
    C2=lambda n: 1e-2,
    C3=lambda n: 1e-3,
    C4=lambda n: 1e-4,
    D0=diminish0,
    D1=diminish1,
    D2=diminish2,
    No=lambda n: 0.,
)
