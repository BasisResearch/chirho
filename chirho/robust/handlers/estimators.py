from typing import TypeVar

from typing_extensions import ParamSpec

from chirho.robust.ops import Functional, Point, influence_fn

P = ParamSpec("P")
S = TypeVar("S")
T = TypeVar("T")


def one_step_correction(
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
    return influence_fn(functional, *test_points, **influence_kwargs_one_step)
