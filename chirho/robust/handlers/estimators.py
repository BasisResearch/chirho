from typing import Any, Callable, Optional

import pyro
import torch
from pyro.infer.elbo import ELBO
from typing_extensions import Concatenate

from chirho.robust.ops import Functional, P, Point, S, T, influence_fn


def tmle(
    model: Callable[P, Any],
    guide: Callable[P, Any],
    functional: Optional[Functional[P, S]] = None,
    *,
    learning_rate: float = 0.01,
    n_steps: int = 10000,
    n_tmle_steps: int = 1,
    num_samples: int = 1000,
    **influence_kwargs,
) -> Callable[Concatenate[Point[T], P], S]:
    from chirho.robust.internals.predictive import (
        BatchedNMCLogPredictiveLikelihood,
    )

    def _corrected_negative_log_likelihood(
        flat_epsilon, flat_influence, plug_in_log_likelihood
    ):
        log_likelihood_correction = torch.log(
            1
            + sum(
                [
                    (influ * eps).sum(-1)
                    for influ, eps in zip(flat_influence, flat_epsilon)
                ]
            )
        )
        return -torch.sum(log_likelihood_correction + plug_in_log_likelihood)

    def _solve_epsilon(test_data: Point[T], prev_guide: Callable[P, Any], *args, **kwargs) -> S:
        # find epsilon that minimizes the corrected density on test data

        influence_at_test = influence_fn(model, prev_guide, functional, **influence_kwargs)(
            test_data, *args, **kwargs
        )

        flat_influence_at_test, treespec = torch.utils._pytree.tree_flatten(
            influence_at_test
        )

        # TODO: Probably remove this?
        # flat_influence_at_test = [inf.detach() for inf in flat_influence_at_test]

        plug_in_log_likelihood_at_test = BatchedNMCLogPredictiveLikelihood(
            model, prev_guide
        )(test_data, *args, **kwargs).detach()

        grad_fn = torch.func.grad(_corrected_negative_log_likelihood)

        # Initialize flat epsilon to be zeros of the same shape as the influence_at_test, excluding any leftmost dimensions.
        flat_epsilon = [
            torch.zeros(i.shape[1:], requires_grad=True) for i in flat_influence_at_test
        ]

        for _ in range(n_steps):
            # Maximum likelihood estimation over epsilon
            grad = grad_fn(
                flat_epsilon, flat_influence_at_test, plug_in_log_likelihood_at_test
            )
            flat_epsilon = [
                eps - learning_rate * g for eps, g in zip(flat_epsilon, grad)
            ]

        return torch.utils._pytree.tree_unflatten(flat_epsilon, treespec)

    def _solve_guide_projection(epsilon: Point[T], prev_guide: Callable[P, Any], *args, **kwargs) -> Callable[P, Any]:
        # TODO: hope this works ... copying parameters in global scope sounds fishy
        new_guide = prev_guide.copy()

        # TODO ...

        return new_guide

    def _one_step(test_data: Point[T], prev_guide: Callable[P, Any], *args, **kwargs) -> Callable[P, Any]:
        # TODO: assert that this does not have side effects on prev_guide

        epsilon = _solve_epsilon(test_data, prev_guide, *args, **kwargs)

        new_guide = _solve_guide_projection(epsilon, prev_guide, *args, **kwargs)

        return new_guide
    
    def _tmle(test_data: Point[T], *args, **kwargs) -> S:
        # TODO: hope this works ... copying parameters in global scope sounds fishy
        tmle_guide = guide

        for _ in range(n_tmle_steps):
            tmle_guide = _one_step(test_data, tmle_guide, *args, **kwargs)
            
        return functional(model, tmle_guide)(*args, **kwargs)
    
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
