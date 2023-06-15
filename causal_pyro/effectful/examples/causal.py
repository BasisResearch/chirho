from typing import Generic, List, Optional, TypeVar

from causal_pyro.effectful.ops.interpretations import StatefulInterpretation, fwd
from causal_pyro.effectful.ops.bootstrap import Operation, define, register

from .forms import Return
from .minipyro import PlateData, enter_plate, exit_plate, plate


T = TypeVar("T")


@define(Operation)
def intervene(obs: T, act: Optional[T] = None) -> T:
    return act if act is not None else obs


class MultiWorldCounterfactual(Generic[T], StatefulInterpretation[List[PlateData], T]):
    state: List[PlateData]


@register(intervene, MultiWorldCounterfactual)
def multi_world_counterfactual_intervene(
    cf_plates: List[PlateData],
    result: Optional[T],
    obs: T,
    act: Optional[T] = None
) -> T:
    act_result = fwd(result)
    new_plate = enter_plate(plate("__intervention__", 2))
    cf_plates.append(new_plate)
    return scatter(obs, act_result, dim=new_plate.dim)


@register(Return, MultiWorldCounterfactual)
def multi_world_counterfactual_return(
    cf_plates: List[PlateData],
    result: Optional[T],
    value: Optional[T],
) -> T:
    while cf_plates:
        exit_plate(cf_plates.pop())
    result = fwd(result)
    return result
