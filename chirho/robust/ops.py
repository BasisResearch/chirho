import functools
from typing import Any, Callable, Mapping, Optional, TypeVar

import torch
from typing_extensions import Concatenate, ParamSpec

from chirho.observational.ops import Observation

P = ParamSpec("P")
Q = ParamSpec("Q")
S = TypeVar("S")
T = TypeVar("T")

Point = Mapping[str, Observation[T]]
Functional = Callable[[Callable[P, Any], Callable[P, Any]], Callable[P, S]]


def influence_fn(
    model: Callable[P, Any],
    guide: Callable[P, Any],
    functional: Optional[Functional[P, S]] = None,
    **linearize_kwargs
) -> Callable[Concatenate[Point[T], bool, P], S]:
    from chirho.robust.internals.linearize import linearize
    from chirho.robust.internals.predictive import PredictiveFunctional
    from chirho.robust.internals.utils import make_functional_call

    linearized = linearize(model, guide, **linearize_kwargs)

    if functional is None:
        assert isinstance(model, torch.nn.Module)
        assert isinstance(guide, torch.nn.Module)
        target = PredictiveFunctional(model, guide)
    else:
        target = functional(model, guide)

    # TODO check that target_params == model_params | guide_params
    assert isinstance(target, torch.nn.Module)
    target_params, func_target = make_functional_call(target)

    @functools.wraps(target)
    def _fn(
        points: Point[T],
        pointwise_influence: bool = False,
        *args: P.args,
        **kwargs: P.kwargs
    ) -> S:
        param_eif = linearized(
            points, pointwise_influence=pointwise_influence, *args, **kwargs
        )
        return torch.vmap(
            lambda d: torch.func.jvp(
                lambda p: func_target(p, *args, **kwargs), (target_params,), (d,)
            )[1],
            in_dims=0,
            randomness="different",
        )(param_eif)

    return _fn
