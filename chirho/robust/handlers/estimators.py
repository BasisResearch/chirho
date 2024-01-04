from typing import Any, Callable, Optional

from typing_extensions import Concatenate

from chirho.robust.ops import Functional, P, Point, S, T, influence_fn


def one_step_correction(
    model: Callable[P, Any],
    guide: Callable[P, Any],
    functional: Optional[Functional[P, S]] = None,
    influence_fn_estimator: Callable[
        [Callable[P, Any], Callable[P, Any], Optional[Functional[P, S]]],
        Callable[Concatenate[Point[T], P], S],
    ] = influence_fn,
    **influence_kwargs,
) -> Callable[Concatenate[Point[T], P], S]:
    """
    Returns a function that computes the one-step correction for the
    functional at a specified set of test points as discussed in
    [1].

    :param model: Python callable containing Pyro primitives.
    :type model: Callable[P, Any]
    :param guide: Python callable containing Pyro primitives.
    :type guide: Callable[P, Any]
    :param functional: model summary of interest, which is a function of the
        model and guide. If ``None``, defaults to :class:`PredictiveFunctional`.
    :type functional: Optional[Functional[P, S]], optional
    :param influence_fn_estimator: function to approximate the efficient influence
        function. Defaults to :func:`influence_fn`.
    :type influence_fn_estimator: Callable[
        [Callable[P, Any], Callable[P, Any], Optional[Functional[P, S]]],
        Callable[Concatenate[Point[T], P], S],
    ]
    :return: function to compute the one-step correction
    :rtype: Callable[Concatenate[Point[T], P], S]

    **References**

    [1] `Semiparametric doubly robust targeted double machine learning: a review`,
    Edward H. Kennedy, 2022.
    """
    influence_kwargs_one_step = influence_kwargs.copy()
    influence_kwargs_one_step["pointwise_influence"] = False
    eif_fn = influence_fn_estimator(
        model, guide, functional, **influence_kwargs_one_step
    )

    def _one_step(test_data: Point[T], *args, **kwargs) -> S:
        return eif_fn(test_data, *args, **kwargs)

    return _one_step


def one_step_corrected_estimator(
    model: Callable[P, Any],
    guide: Callable[P, Any],
    functional: Optional[Functional[P, S]] = None,
    influence_fn_estimator=influence_fn,
    **influence_kwargs,
) -> Callable[Concatenate[Point[T], P], S]:
    """
    Returns a function that computes the one-step corrected estimator for the functional at a
    specified set of test points as discussed in [1].

    :param model: Python callable containing Pyro primitives.
    :type model: Callable[P, Any]
    :param guide: Python callable containing Pyro primitives.
        Must only contain continuous latent variables.
    :type guide: Callable[P, Any]
    :param functional: model summary of interest, which is a function of the
        model and guide. If ``None``, defaults to :class:`PredictiveFunctional`.
    :type functional: Optional[Functional[P, S]], optional
    :param influence_fn_estimator: function to approximate the efficient influence
        function. Defaults to :func:`influence_fn`.
    :type influence_fn_estimator: Callable[
        [Callable[P, Any], Callable[P, Any], Optional[Functional[P, S]]],
        Callable[Concatenate[Point[T], P], S],
    ]
    :return: function to computer the one-step corrected estimator
    :rtype: S
    """
    plug_in_estimator = functional(model, guide)
    correction = one_step_correction(
        model, guide, functional, influence_fn_estimator, **influence_kwargs
    )

    def _one_step_corrected_estimator(test_data: Point[T], *args, **kwargs) -> S:
        return plug_in_estimator(*args, **kwargs) + correction(
            test_data, *args, **kwargs
        )

    return _one_step_corrected_estimator
