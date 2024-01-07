from typing import Any, Callable, Optional

import torch
from typing_extensions import Concatenate

from chirho.robust.ops import Functional, P, Point, S, T, influence_fn


def tmle(
    model: Callable[P, Any],
    guide: Callable[P, Any],
    functional: Optional[Functional[P, S]] = None,
    **influence_kwargs,
) -> Callable[Concatenate[Point[T], P], S]:
    from chirho.robust.internals.predictive import BatchedNMCLogPredictiveLikelihood

    eif_fn = influence_fn(model, guide, functional, **influence_kwargs)

    # TODO: detach non-optimizable tensors in model and guide for optimization of epsilon.

    def _initialize_epsilon(test_data: Point[T], *args, **kwargs) -> Point[T]:
        correction = eif_fn(test_data, *args, **kwargs)
        flat_correction, treespec = torch.utils._pytree.tree_flatten(correction)
        return torch.utils._pytree.tree_unflatten(
            [torch.ones(c.shape[1:]) for c in flat_correction], treespec
        )

    def _corrected_log_likelihood_estimator(
        test_data: Point[T],
        epsilon: Point[T] = None,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        plug_in_log_likelihood = BatchedNMCLogPredictiveLikelihood(model, guide)(
            test_data, *args, **kwargs
        )

        correction = eif_fn(test_data, *args, **kwargs)

        flat_correction, _ = torch.utils._pytree.tree_flatten(correction)
        flat_epsilon, _ = torch.utils._pytree.tree_flatten(epsilon)

        return (
            torch.log(
                1
                + torch.sum(
                    [
                        correction * epsilon
                        for correction, epsilon in zip(flat_correction, flat_epsilon)
                    ]
                )
            )
            + plug_in_log_likelihood
        )

        # return torch.log(1 + flat_epsilon * flat_correction) + plug_in_log_likelihood

    def _tmle(test_data: Point[T], *args, **kwargs) -> S:
        epsilon = _initialize_epsilon(test_data, *args, **kwargs)

        corrected_log_likelihood = _corrected_log_likelihood_estimator(
            test_data, epsilon, *args, **kwargs
        )

        # TODO: optimize corrected_log_likelihood
        pass

    return _tmle


def one_step_correction(
    model: Callable[P, Any],
    guide: Callable[P, Any],
    functional: Optional[Functional[P, S]] = None,
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
    :return: function to compute the one-step correction
    :rtype: Callable[Concatenate[Point[T], P], S]

    **References**

    [1] `Semiparametric doubly robust targeted double machine learning: a review`,
    Edward H. Kennedy, 2022.
    """
    influence_kwargs_one_step = influence_kwargs.copy()
    influence_kwargs_one_step["pointwise_influence"] = False
    eif_fn = influence_fn(model, guide, functional, **influence_kwargs_one_step)

    def _one_step(test_data: Point[T], *args, **kwargs) -> S:
        return eif_fn(test_data, *args, **kwargs)

    return _one_step
