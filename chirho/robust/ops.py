import collections
from functools import partial
from typing import Callable, Dict, List, Optional
import torch
import pyro
from pyro.nn import PyroModule
from torch.autograd.functional import jacobian

from chirho.observational.handlers import condition
from chirho.robust.utils import (
    _flatten_dict,
    _unflatten_dict,
    _conjugate_gradient,
    _check_correct_model_signature,
    _check_correct_functional_signature,
)


def _make_fisher_jvp(f: Callable, theta: torch.tensor, n_monte_carlo: int) -> Callable:
    r"""
    Args:
        f (Callable): The log-likelihood function to differentiate.
        theta (torch.tensor): Model parameters.
        n_monte_carlo (int): The number of Monte Carlo samples to use.
    Returns:
        Callable: A function that computes the empirical Fisher information x vector.
    """

    def empirical_fisher_vp(v):
        vnew = torch.func.jvp(f, (theta,), (v / n_monte_carlo,))[1]
        (_, vjpfunc) = torch.func.vjp(f, theta)
        return vjpfunc(vnew)[0]

    return empirical_fisher_vp


def empirical_inverse_fisher_vp(
    f: Callable,
    theta: torch.tensor,
    v: torch.tensor,
    X: Dict[str, torch.tensor],
    *,
    cg_iters: int,
) -> torch.tensor:
    r"""
    Args:
        f (Callable): Log-likelihood function to differentiate, which maps theta to
            :math:`\log P(X | \theta) \in \mathbb{R}^{\text{n_monte_carlo}}`.
        theta (torch.tensor): Model parameters.
        v (torch.tensor): The vector to multiply by the inverse Fisher information.
        X (Dict[str, torch.tensor]): Data simulated from the model with parameters equal to theta.
            Consists of `n_monte_carlo` total samples.
        cg_iters (int): The number of conjugate gradient iterations to use.
    Returns:
        torch.tensor: The product of the inverse Fisher information and v.
    """
    f_Ax = _make_fisher_jvp(partial(f, X=X), theta, X[next(iter(X))].shape[0])
    return _conjugate_gradient(f_Ax, v, cg_iters)


def one_step_correction(
    model: PyroModule,
    theta_hat: Dict[str, torch.tensor],
    target_functional: Callable[[Callable], torch.tensor],
    X_test: Dict[str, torch.tensor],
    *,
    pointwise_influence: bool = False,
    n_monte_carlo: int = None,
    cg_iters: int = 1000,
    obs_names: Optional[List[str]] = None,
) -> torch.tensor:
    r"""
    One step correction for a given target functional.

    Args:
        model (PyroModule): Pyro model.
        theta_hat (Dict[str, torch.tensor]): The parameters to condition on.
        target_functional (Callable): The functional to compute the one step correction for.
        X_test (Dict[str, torch.tensor]): The test data.
        pointwise_influence (bool): If True, compute the one step correction for each data point in `X_test`.
            Otherwise, compute the one step correction averaged over `X_test`
            (which is the final one step correction estimator).
        n_monte_carlo (int): The number of Monte Carlo samples to use when approximating fisher information matrix.
            If None, this defaults to 25 * dim(`theta_hat`).
        cg_iters (int): The number of conjugate gradient iterations to use for inverting fisher information matrix.
        obs_names (List[str]): The names of the observed variables in the model. If None, defaults to the keys in
            `X_test`.

    Returns:
        torch.tensor: The one step correction for the given target functional (if `pointwise_influence` = False).
            Otherwise, outputs the efficient influence function (approximated via Monte Carlo) evaluated at each
            data point in `X_test`.
    """
    _check_correct_model_signature(model)
    _check_correct_functional_signature(target_functional)

    # Canonical ordering of parameters when flattening and unflattening
    theta_hat = collections.OrderedDict(
        (k, theta_hat[k]) for k in sorted(theta_hat.keys())
    )
    flat_theta = _flatten_dict(theta_hat)
    theta_dim = flat_theta.shape[0]

    if n_monte_carlo is None:
        n_monte_carlo = 25 * theta_dim  # 25 samples per parameter
    else:
        assert (
            n_monte_carlo >= theta_dim
        ), "n_monte_carlo must be at least as large as the number of parameters to invert empirical fisher matrix."
        if n_monte_carlo < 25 * theta_dim:
            print(
                "Warning: n_monte_carlo is less than 25x # of parameters, which may lead to inaccurate estimates."
            )

    # Compute gradient of plug-in functional
    plug_in = target_functional(model, theta_hat)
    plug_in += 0 * flat_theta.sum()  # hack for full gradient (maintain flattened shape)

    plug_in_grads = _flatten_dict(
        collections.OrderedDict(
            zip(
                theta_hat.keys(),
                torch.autograd.grad(plug_in, theta_hat.values()),
            )
        )
    )

    # Compute gradient of log P(X_test | theta_hat)
    if obs_names is None:
        obs_names = list(X_test.keys())

    log_likelihood_fn = make_log_likelihood_fn(model, theta_hat, obs_names)
    log_likelihood_fn_at_test = partial(log_likelihood_fn, X=X_test)
    scores_test = jacobian(log_likelihood_fn_at_test, flat_theta)

    assert scores_test.shape[0] == X_test[next(iter(X_test))].shape[0]
    assert scores_test.shape[1] == flat_theta.shape[0]

    if not pointwise_influence:
        scores_test = scores_test.mean(dim=0).unsqueeze(0)  # 1 x dim(theta_hat)

    # Compute inverse fisher information x scores_test, where
    # fisher information is approximated by simulating data from the model
    efficient_influence_fn_vals = torch.zeros(scores_test.shape[0])
    sim_data = simulate_data_from_model(model, theta_hat, n_monte_carlo, obs_names)

    # TODO: use functorch to vectorize this loop. Current for loop severely slows method down
    # if `pointwise_influence` = True and `X_test` is large.
    for j, score in enumerate(scores_test):
        inv_fish_score = empirical_inverse_fisher_vp(
            log_likelihood_fn, flat_theta, score, sim_data, cg_iters=cg_iters
        )
        efficient_influence_fn_vals[j] = plug_in_grads.dot(inv_fish_score).item()

    return efficient_influence_fn_vals


def bayesian_one_step_correction():
    # TODO: implement Bayesian one step correction using
    # `one_step_correction` in Algorithm 1 of
    # https://arxiv.org/abs/2306.06059
    raise NotImplementedError


def simulate_data_from_model(
    model: PyroModule,
    theta_hat: Dict[str, torch.tensor],
    n_monte_carlo: int,
    obs_names: Optional[List[str]] = None,
):
    r"""
    Simulate data from `model` with parameters set equal to `theta_hat`.
    """
    model_at_theta = condition(data=theta_hat)(model)
    with pyro.poutine.trace() as model_tr:
        model_at_theta(N=n_monte_carlo)
    if obs_names is None:
        return {
            k: model_tr.trace.nodes[k]["value"]
            for k in model_tr.trace.nodes.keys()
            if "value" in model_tr.trace.nodes[k]
        }
    else:
        return {k: model_tr.trace.nodes[k]["value"] for k in obs_names}


def make_log_likelihood_fn(
    model: PyroModule,
    theta_hat: Dict[str, torch.tensor],
    obs_names: List[str],
) -> torch.tensor:
    r"""
    Args:
        model (PyroModule): Pyro model.
        theta_hat (Dict[str, torch.tensor]): The parameters to condition on.
            This is only used to order the dictionary when unflattening parameters in the function below.
        obs_names (List[str]): The names of the observed variables in the model.
    Returns:
        Callable: A function that computes the log likelihood of the model at each data point in `X`. This function
            maps `theta`, `X` to :math:`\log P(X | \theta) \in \mathbb{R}^{\text{n_monte_carlo}}`.
    """

    def log_likelihood_fn(flat_theta: torch.tensor, X: Dict[str, torch.Tensor]):
        n_monte_carlo = X[next(iter(X))].shape[0]
        theta = _unflatten_dict(flat_theta, theta_hat)
        model_at_theta = condition(data=theta)(DataConditionedModel(model))
        log_like_trace = pyro.poutine.trace(model_at_theta).get_trace(X)
        log_like_trace.compute_log_prob()
        log_prob_at_datapoints = torch.zeros(n_monte_carlo)
        for name in obs_names:
            log_prob_at_datapoints += log_like_trace.nodes[name]["log_prob"]
        return log_prob_at_datapoints

    return log_likelihood_fn


class DataConditionedModel(PyroModule):
    r"""
    Helper class for conditioning on data.
    """

    def __init__(self, model: PyroModule):
        super().__init__()
        self.model = model

    def forward(self, D: Dict[str, torch.tensor]):
        with condition(data=D):
            # Assume first dimension corresponds to # of datapoints
            N = D[next(iter(D))].shape[0]
            return self.model.forward(N=N)
