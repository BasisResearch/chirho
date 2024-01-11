import copy
from typing import Any, Callable, List, TypeVar

import pyro
import torch
from typing_extensions import ParamSpec

from chirho.robust.internals.utils import make_functional_call
from chirho.robust.ops import Functional, Point, influence_fn

P = ParamSpec("P")
S = TypeVar("S")
T = TypeVar("T")


def tmle(
    model: Callable[P, Any],
    functional: Functional[P, S],
    *,
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

    def _solve_epsilon(
        test_data: Point[T], prev_model: Callable[P, Any], *args, **kwargs
    ) -> S:
        # find epsilon that minimizes the corrected density on test data

        influence_at_test = influence_fn(prev_model, functional, **influence_kwargs)(
            test_data, *args, **kwargs
        )

        flat_influence_at_test, treespec = torch.utils._pytree.tree_flatten(
            influence_at_test
        )

        # TODO: Probably remove this?
        # flat_influence_at_test = [inf.detach() for inf in flat_influence_at_test]

        plug_in_log_likelihood_at_test = BatchedNMCLogMarginalLikelihood(prev_model)(
            test_data, *args, **kwargs
        ).detach()

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

    def _solve_model_projection(
        epsilon: Point[T],
        prev_model: Callable[P, Any],
        obs_names: List[str],
        *args,
        **kwargs,
    ) -> Callable[P, Any]:
        batched_log_prob = BatchedNMCLogMarginalLikelihood(
            prev_model, num_samples=num_samples
        )
        prev_params, log_p_phi = make_functional_call(batched_log_prob)
        prev_params, functional_model = make_functional_call(prev_model)

        new_params = copy.deepcopy(prev_params)

        flat_epsilon, _ = torch.utils._pytree.tree_flatten(epsilon)

        def log_p_epsilon(params, x, *args, **kwargs):
            influence_at_x = influence_fn(prev_model, functional, **influence_kwargs)(
                x, *args, **kwargs
            )
            flat_influence_at_x, _ = torch.utils._pytree.tree_flatten(influence_at_x)

            plug_in_log_likelihood_at_x = log_p_phi(params, x, *args, **kwargs)

            return _corrected_negative_log_likelihood(
                flat_epsilon, flat_influence_at_x, plug_in_log_likelihood_at_x
            )

        def loss(new_params, *args, **kwargs):
            # Sample data from the variational approximation
            with pyro.poutine.trace() as tr:
                functional_model(new_params, *args, **kwargs)

            samples = {
                k: v["value"] for k, v in tr.trace.nodes.items() if k in obs_names
            }

            a = batched_log_prob

            assert False

            return log_p_phi(new_params, samples, *args, **kwargs) - log_p_epsilon(
                new_params, samples, *args, **kwargs
            )

        test_loss = loss(new_params, *args, **kwargs)

        assert False

        return new_model

    def _one_step(
        test_data: Point[T], prev_model: Callable[P, Any], *args, **kwargs
    ) -> Callable[P, Any]:
        # TODO: assert that this does not have side effects on prev_model

        epsilon = _solve_epsilon(test_data, prev_model, *args, **kwargs)

        new_model = _solve_model_projection(
            epsilon, prev_model, list(test_data.keys()), *args, **kwargs
        )

        return new_model

    def _tmle(test_data: Point[T], *args, **kwargs) -> S:
        # TODO: hope this works ... copying parameters in global scope sounds fishy
        tmle_model = model

        for _ in range(n_tmle_steps):
            tmle_model = _one_step(test_data, tmle_model, *args, **kwargs)

        return functional(tmle_model)(*args, **kwargs)

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
