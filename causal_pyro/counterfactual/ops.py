from typing import List, Optional, Tuple, TypeVar

import pyro

from causal_pyro.indexed.ops import IndexSet, scatter
from causal_pyro.interventional.ops import Intervention, intervene

T = TypeVar("T")


@pyro.poutine.runtime.effectful(type="gen_intervene_name")
def gen_intervene_name(name: Optional[str] = None) -> str:
    if name is not None:
        return name
    raise NotImplementedError


@pyro.poutine.runtime.effectful(type="split")
@pyro.poutine.block(hide_types=["intervene"])
def split(
    obs: T,
    acts: Tuple[Intervention[T], ...],
    *,
    name: Optional[str] = None,
    **kwargs
) -> T:
    """
    Split the state of the world at an intervention.
    """
    name = gen_intervene_name(name)

    act_values: List[Tuple[IndexSet, T]] = [(IndexSet(**{name: 0}), obs)]
    for i, act in enumerate(acts):
        act_values += [
            (IndexSet(**{name: i + 1}), intervene(act_values[i][1], act, **kwargs))
        ]

    return scatter(dict(act_values), event_dim=kwargs.get("event_dim", 0))
