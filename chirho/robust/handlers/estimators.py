import copy
from typing import Any, Callable, TypeVar

import pyro
import torch
from typing_extensions import ParamSpec

from chirho.robust.handlers.predictive import PredictiveFunctional
from chirho.robust.internals.utils import make_functional_call
from chirho.robust.ops import Functional, Point, influence_fn

P = ParamSpec("P")
S = TypeVar("S")
T = TypeVar("T")


def tmle(
    functional: Functional[P, S],
    test_point: Point,
    learning_rate: float = 0.01,
    n_steps: int = 10000,
    n_tmle_steps: int = 1,
    num_samples: int = 1000,
    **influence_kwargs,
) -> Functional[P, S]:
    from chirho.robust.internals.nmc import BatchedNMCLogMarginalLikelihood

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

    def loss_for_epsilon(
        flat_epsilon,
        flat_influence,
        plug_in_log_likelihood,
        penalty_strength: float = 1e6,
    ):
        term = 1 + sum(
            [(influ * eps).sum(-1) for influ, eps in zip(flat_influence, flat_epsilon)]
        )

        penalty = penalty_strength * torch.relu(-1 * term).sum()
        penalty2 = penalty_strength * torch.relu(-1 * term).sum()

        if penalty > 0:
            import pdb

            pdb.set_trace()
        return (
            _corrected_negative_log_likelihood(
                flat_epsilon, flat_influence, plug_in_log_likelihood
            )
            + penalty
        )

    def _solve_epsilon(prev_model: Callable[P, Any], *args, **kwargs) -> S:
        # find epsilon that minimizes the corrected density on test data

        influence_at_test = influence_fn(functional, test_point, **influence_kwargs)(
            prev_model
        )(*args, **kwargs)

        flat_influence_at_test, treespec = torch.utils._pytree.tree_flatten(
            influence_at_test
        )

        import pdb

        pdb.set_trace()

        # TODO: Probably remove this?
        # flat_influence_at_test = [inf.detach() for inf in flat_influence_at_test]

        plug_in_log_likelihood_at_test = BatchedNMCLogMarginalLikelihood(prev_model)(
            test_point, *args, **kwargs
        ).detach()

        grad_fn = torch.func.grad(loss_for_epsilon)

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

    def _solve_model_projection(
        epsilon: Point[T],
        prev_model: Callable[P, Any],
        *args,
        **kwargs,
    ) -> Callable[P, Any]:
        batched_log_prob = BatchedNMCLogMarginalLikelihood(
            prev_model, num_samples=num_samples
        )
        prev_params, log_p_phi = make_functional_call(batched_log_prob)
        _, functional_model = make_functional_call(PredictiveFunctional(prev_model))

        new_params = copy.deepcopy(prev_params)

        flat_epsilon, _ = torch.utils._pytree.tree_flatten(epsilon)

        def log_p_epsilon(params, x, *args, **kwargs):
            influence_at_x = influence_fn(functional, x, **influence_kwargs)(
                prev_model
            )(*args, **kwargs)
            flat_influence_at_x, _ = torch.utils._pytree.tree_flatten(influence_at_x)

            plug_in_log_likelihood_at_x = log_p_phi(params, x, *args, **kwargs)

            # import pdb

            # pdb.set_trace()

            return -1 * _corrected_negative_log_likelihood(
                flat_epsilon, flat_influence_at_x, plug_in_log_likelihood_at_x
            )

        def loss(new_params, *args, **kwargs):
            # Sample data from the variational approximation
            samples = {
                k: v
                for k, v in functional_model(new_params, *args, **kwargs).items()
                if k in test_point
            }
            term1 = log_p_phi(new_params, samples, *args, **kwargs)
            term2 = log_p_epsilon(new_params, samples, *args, **kwargs)
            import pdb

            pdb.set_trace()
            return torch.sum(term1) - term2

        test_loss = loss(new_params, *args, **kwargs)

        assert False

        return new_model

    def _one_step(prev_model: Callable[P, Any], *args, **kwargs) -> Callable[P, Any]:
        # TODO: assert that this does not have side effects on prev_model
        epsilon = _solve_epsilon(prev_model, *args, **kwargs)

        new_model = _solve_model_projection(epsilon, prev_model, *args, **kwargs)

        return new_model

    def _tmle(model: Callable[P, Any]) -> S:
        def _tmle_inner(*args, **kwargs):
            # TODO: hope this works ... copying parameters in global scope sounds fishy
            tmle_model = model

            for _ in range(n_tmle_steps):
                tmle_model = _one_step(tmle_model, *args, **kwargs)

            return functional(tmle_model)(*args, **kwargs)

        return _tmle_inner

    return _tmle


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