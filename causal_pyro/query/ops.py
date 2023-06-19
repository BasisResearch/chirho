from typing import Callable, Dict, TypeVar

import pyro

T = TypeVar("T")


@pyro.poutine.runtime.effectful(type="expectation")
def expectation(
    model: Callable[..., T],
    name: str,
    # added axis here (?)
    model_args,
    axis: int = 0,
    model_kwargs: Dict = {},
    *args,
    **kwargs
):
    raise NotImplementedError(
        "The expectation operation requires an approximation method \
        to be evaluated. Consider wrapping in a handler found \
        in `causal_pyro.query.handlers`."
    )
