from typing import List, Optional, TypeVar

from ..ops.interpretations import Interpretation, fwd
from ..ops.terms import Environment, Operation, define

from .forms import Return
from .minipyro import PlateData, enter_plate, exit_plate, plate


T = TypeVar("T")


@define(Operation)
def intervene(obs: T, act: Optional[T] = None) -> T:
    return act if act is not None else obs


class MultiWorldCounterfactual(Interpretation[List[PlateData]]):
    state: List[PlateData]


@MultiWorldCounterfactual.union_(intervene)
def multi_world_counterfactual_intervene(
    cf_plates: List[PlateData],
    ctx: Environment[T],
    result: Optional[T],
    obs: T,
    act: Optional[T] = None
) -> T:
    act_result = fwd(ctx, result)
    new_plate = enter_plate(plate("__intervention__", 2))
    cf_plates.append(new_plate)
    return scatter(obs, act_result, dim=new_plate.dim)


@MultiWorldCounterfactual.union_(Return)
def multi_world_counterfactual_return(
    cf_plates: List[PlateData],
    ctx: Environment[T],
    result: Optional[T],
    value: Optional[T],
) -> T:
    while cf_plates:
        exit_plate(cf_plates.pop())
    result = fwd(ctx, result)
    return result
