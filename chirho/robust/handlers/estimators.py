from typing import Any, Callable, TypeVar

import torch
from typing_extensions import ParamSpec

from chirho.robust.ops import Functional, Point, influence_fn

P = ParamSpec("P")
S = TypeVar("S")
T = TypeVar("T")


def one_step_corrected_estimator(
    functional: Functional[P, S],
    *test_points: Point[T],
    **influence_kwargs,
) -> Functional[P, S]:
    """
    Returns a functional that computes the one-step correction for the
    functional at a specified set of test points as discussed in [1].

    :param functional: model summary functional of interest
    :param test_points: points at which to compute the one-step correction
    :return: functional to compute the one-step correction

    **References**

    [1] `Semiparametric doubly robust targeted double machine learning: a review`,
    Edward H. Kennedy, 2022.
    """
    influence_kwargs_one_step = influence_kwargs.copy()
    influence_kwargs_one_step["pointwise_influence"] = False
    eif_fn = influence_fn(functional, *test_points, **influence_kwargs_one_step)

    def _corrected_functional(*model: Callable[P, Any]) -> Callable[P, S]:
        plug_in_estimator = functional(*model)
        correction_estimator = eif_fn(*model)

        def _estimator(*args, **kwargs) -> S:
            plug_in_estimate = plug_in_estimator(*args, **kwargs)
            correction = correction_estimator(*args, **kwargs)

            flat_plug_in_estimate, treespec = torch.utils._pytree.tree_flatten(
                plug_in_estimate
            )
            flat_correction, _ = torch.utils._pytree.tree_flatten(correction)

            return torch.utils._pytree.tree_unflatten(
                [a + b for a, b in zip(flat_plug_in_estimate, flat_correction)],
                treespec,
            )

        return _estimator

    return _corrected_functional
