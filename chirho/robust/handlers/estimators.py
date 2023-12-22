from typing import Any, Callable, Concatenate, Optional

from chirho.robust.ops import Functional, P, Point, S, T, influence_fn


def one_step_correction(
    model: Callable[P, Any],
    guide: Callable[P, Any],
    functional: Optional[Functional[P, S]] = None,
    **influence_kwargs,
) -> Callable[Concatenate[Point[T], P], S]:
    influence_kwargs_one_step = influence_kwargs.copy()
    influence_kwargs_one_step["pointwise_influence"] = False
    eif_fn = influence_fn(model, guide, functional, **influence_kwargs_one_step)

    def _one_step(test_data: Point[T], *args, **kwargs) -> S:
        return eif_fn(test_data, *args, **kwargs)

    return _one_step
