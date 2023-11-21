import functools
from typing import (
    Any,
    Callable,
    Concatenate,
    Mapping,
    Optional,
    ParamSpec,
    Tuple,
    TypeVar,
)

import pyro
import torch
from chirho.observational.ops import Observation

P = ParamSpec("P")
Q = ParamSpec("Q")
S = TypeVar("S")
T = TypeVar("T")

Model = Callable[P, Any]
Point = Mapping[str, Observation[T]]
Functional = Callable[[Model[P], Model[P]], Callable[P, S]]
ParamDict = Mapping[str, torch.Tensor]


def make_functional_call(
    mod: Callable[P, T]
) -> Tuple[ParamDict, Callable[Concatenate[ParamDict, P], T]]:
    assert isinstance(mod, torch.nn.Module)
    param_dict: ParamDict = dict(mod.named_parameters())
    mod_func: Callable[Concatenate[ParamDict, P], T] = functools.partial(torch.func.functional_call, mod)
    functionalized_mod_func: Callable[Concatenate[ParamDict, P], T] = torch.func.functionalize(
        pyro.validation_enabled(False)(mod_func)
    )
    return param_dict, functionalized_mod_func


def influence_fn(
    model: Model[P],
    guide: Model[P],
    functional: Optional[Functional[P, S]] = None,
    **linearize_kwargs
) -> Callable[Concatenate[Point[T], P], S]:

    from chirho.robust.internals import linearize

    linearized = linearize(model, guide, **linearize_kwargs)

    if functional is None:
        return linearized

    target = functional(model, guide)
    assert isinstance(target, torch.nn.Module)
    target_params, func_target = make_functional_call(target)

    @functools.wraps(target)
    def _fn(point: Point[T], *args: P.args, **kwargs: P.kwargs) -> S:
        param_eif = linearized(point, *args, **kwargs)
        return torch.func.jvp(
            lambda p: func_target(p, *args, **kwargs),
            (target_params,),
            (param_eif,)
        )[1]

    return _fn
