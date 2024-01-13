import copy
import pdb
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
    learning_rate: float = 1e5,
    n_steps: int = 100,
    n_tmle_steps: int = 1,
    num_nmc_samples: int = 1000,
    num_grad_samples: int = 100,
    log_jitter: float = 1e-6,
    **influence_kwargs,
) -> Functional[P, S]:
    from chirho.robust.internals.nmc import BatchedNMCLogMarginalLikelihood

    def tmle_scipy_optimize_wrapper(packed_influence) -> torch.Tensor:
        import numpy as np
        import scipy
        from scipy.optimize import LinearConstraint
        # Turn things into numpy. This makes us sad... :(
        D = packed_influence.detach().numpy()

        N, L = D.shape[0], D.shape[1]

        def loss(epsilon):
            correction = 1 + D.dot(epsilon)

            return np.sum(np.log(np.maximum(correction, log_jitter)))
        
        positive_density_constraint = LinearConstraint(D, -1 * np.ones(N), np.inf * np.ones(N))

        epsilon_solve = scipy.optimize.minimize(loss, np.zeros(L, dtype=D.dtype), constraints=positive_density_constraint)

        if not epsilon_solve.success:
            raise RuntimeError("TMLE optimization did not converge.")
        
        # Convert epsilon back to torch. This makes us happy... :)
        packed_epsilon = torch.tensor(epsilon_solve.x, dtype=packed_influence.dtype)

        return packed_epsilon

    def _solve_epsilon(prev_model: Callable[P, Any], *args, **kwargs) -> torch.Tensor:
        # find epsilon that minimizes the corrected density on test data

        influence_at_test = influence_fn(functional, test_point, **influence_kwargs)(
            prev_model
        )(*args, **kwargs)

        flat_influence_at_test, _ = torch.utils._pytree.tree_flatten(
            influence_at_test
        )

        N = flat_influence_at_test[0].shape[0]

        packed_influence_at_test = torch.concatenate([i.reshape(N, -1) for i in flat_influence_at_test])

        packed_epsilon = tmle_scipy_optimize_wrapper(packed_influence_at_test)

        return packed_epsilon

    def _solve_model_projection(
        packed_epsilon: torch.Tensor,
        prev_model: Callable[P, Any],
        *args,
        **kwargs,
    ) -> Callable[P, Any]:
        
        batched_log_prob = BatchedNMCLogMarginalLikelihood(
            prev_model, num_samples=num_nmc_samples
        )
        prev_params, log_p_phi_pointwise = make_functional_call(batched_log_prob)
        prev_params = {k: v.detach() for k, v in prev_params.items()}
        
        def log_p_phi(*args, **kwargs) -> torch.Tensor:
            return torch.sum(log_p_phi_pointwise(*args, **kwargs))
        
        _, functional_model = make_functional_call(PredictiveFunctional(prev_model, num_samples=num_grad_samples))

        new_params = copy.deepcopy(prev_params)

        def log_p_epsilon(params, x, *args, **kwargs):
            influence = influence_fn(functional, x, **influence_kwargs)(prev_model)(*args, **kwargs)
            flat_influence, _ = torch.utils._pytree.tree_flatten(influence)
            N_x = flat_influence[0].shape[0]

            packed_influence = torch.concatenate([i.reshape(N_x, -1) for i in flat_influence]).detach()

            log_likelihood_correction = torch.sum(torch.log(torch.maximum(1 + packed_influence.mv(packed_epsilon), torch.tensor(log_jitter))))

            return log_likelihood_correction + log_p_phi(params, x, *args, **kwargs)

        def loss(new_params, *args, **kwargs):
            # Sample data from the variational approximation
            samples = {k: v for k, v in functional_model(new_params, *args, **kwargs).items() if k in test_point}
            term1 = log_p_phi(new_params, samples, *args, **kwargs)
            term2 = log_p_epsilon(new_params, samples, *args, **kwargs)

            return term1 - term2

        grad_fn = torch.func.grad(loss)

        for _ in range(n_steps):
            print(new_params, prev_params, loss(new_params, *args, **kwargs))
            grad = grad_fn(new_params, *args, **kwargs)

            new_params = {k: (v - learning_rate * grad[k]) for k, v in new_params.items()}

        # TODO: pack new_params into a callable

        return new_model

    def _one_step(prev_model: Callable[P, Any], *args, **kwargs) -> Callable[P, Any]:
        # TODO: assert that this does not have side effects on prev_model
        packed_epsilon = _solve_epsilon(prev_model, *args, **kwargs)

        new_model = _solve_model_projection(packed_epsilon, prev_model, *args, **kwargs)

        return new_model

    def _tmle(model: Callable[P, Any]) -> S:
        def _tmle_inner(*args, **kwargs):
            tmle_model = model

            for _ in range(n_tmle_steps):
                tmle_model = _one_step(tmle_model, *args, **kwargs)

            return functional(tmle_model)(*args, **kwargs)

        return _tmle_inner

    return _tmle


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
