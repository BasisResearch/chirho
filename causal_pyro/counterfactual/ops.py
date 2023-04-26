from typing import Optional, Tuple, TypeVar

import pyro

from causal_pyro.interventional.ops import Intervention

T = TypeVar("T")


@pyro.poutine.runtime.effectful(type="gen_intervene_name")
def gen_intervene_name(name: Optional[str] = None) -> str:
    if name is not None:
        return name
    raise NotImplementedError(
        "No handler active for gen_intervene_name. "
        "Did you forget to use MultiWorldCounterfactual?"
    )


@pyro.poutine.runtime.effectful(type="split")
def split(
    obs: T, acts: Tuple[Intervention[T], ...], *, name: Optional[str] = None, **kwargs
) -> T:
    """
    Split the state of the world at an intervention.
    """
    raise NotImplementedError(
        "No handler active for split. "
        "Did you forget to use MultiWorldCounterfactual?"
    )
