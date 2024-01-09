from typing import Any, Callable, TypeVar

from typing_extensions import Concatenate, ParamSpec

from chirho.robust.ops import Functional, Point, influence_fn

P = ParamSpec("P")
S = TypeVar("S")
T = TypeVar("T")


def one_step_correction(
    model: Callable[P, Any],
    functional: Functional[P, S],
    **influence_kwargs,
) -> Callable[Concatenate[Point[T], P], S]:
    """
    Returns a function that computes the one-step correction for the
    functional at a specified set of test points as discussed in
    [1].

    :param model: Python callable containing Pyro primitives.
    :type model: Callable[P, Any]
    :param functional: model summary of interest, which is a function of the model.
    :type functional: Functional[P, S]
    :return: function to compute the one-step correction
    :rtype: Callable[Concatenate[Point[T], P], S]

    **References**

    [1] `Semiparametric doubly robust targeted double machine learning: a review`,
    Edward H. Kennedy, 2022.
    """
    influence_kwargs_one_step = influence_kwargs.copy()
    influence_kwargs_one_step["pointwise_influence"] = False
    eif_fn = influence_fn(model, functional, **influence_kwargs_one_step)

    def _one_step(test_data: Point[T], *args, **kwargs) -> S:
        return eif_fn(test_data, *args, **kwargs)

    return _one_step
