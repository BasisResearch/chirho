import functools
from typing import Any, Callable, Optional, TypeVar

import pyro
import torch
from typing_extensions import Concatenate, ParamSpec

from chirho.robust.internals.predictive import BatchedNMCLogPredictiveLikelihood
from chirho.robust.internals.utils import (
    ParamDict,
    make_flatten_unflatten,
    make_functional_call,
    reset_rng_state,
)
from chirho.robust.ops import Point

P = ParamSpec("P")
Q = ParamSpec("Q")
S = TypeVar("S")
T = TypeVar("T")


def _flat_conjugate_gradient_solve(
    f_Ax: Callable[[torch.Tensor], torch.Tensor],
    b: torch.Tensor,
    *,
    cg_iters: Optional[int] = None,
    residual_tol: float = 1e-3,
) -> torch.Tensor:
    r"""Use Conjugate Gradient iteration to solve Ax = b. Demmel p 312.

    Args:
        f_Ax (callable): A function to compute matrix vector product.
        b (torch.Tensor): Right hand side of the equation to solve.
        cg_iters (int): Number of iterations to run conjugate gradient
            algorithm.
        residual_tol (float): Tolerence for convergence.

    Returns:
        torch.Tensor: Solution x* for equation Ax = b.

    Notes: This code is adapted from
      https://github.com/rlworkgroup/garage/blob/master/src/garage/torch/optimizers/conjugate_gradient_optimizer.py
    """
    assert len(b.shape), "b must be a 2D matrix"

    if cg_iters is None:
        cg_iters = b.shape[1]
    else:
        cg_iters = min(cg_iters, b.shape[1])

    def _batched_dot(x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        return (x1 * x2).sum(axis=-1)  # type: ignore

    def _batched_product(a: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        return a.unsqueeze(0).t() * B

    p = b.clone()
    r = b.clone()
    x = torch.zeros_like(b)
    z = f_Ax(p)
    rdotr = _batched_dot(r, r)
    v = rdotr / _batched_dot(p, z)
    newrdotr = rdotr
    mu = newrdotr / rdotr
    zeros_xr = torch.zeros_like(x)
    for _ in range(cg_iters):
        not_converged = rdotr > residual_tol
        not_converged_broadcasted = not_converged.unsqueeze(0).t()
        z = torch.where(not_converged_broadcasted, f_Ax(p), z)
        v = torch.where(not_converged, rdotr / _batched_dot(p, z), v)
        x += torch.where(not_converged_broadcasted, _batched_product(v, p), zeros_xr)
        r -= torch.where(not_converged_broadcasted, _batched_product(v, z), zeros_xr)
        newrdotr = torch.where(not_converged, _batched_dot(r, r), newrdotr)
        mu = torch.where(not_converged, newrdotr / rdotr, mu)
        p = torch.where(not_converged_broadcasted, r + _batched_product(mu, p), p)
        rdotr = torch.where(not_converged, newrdotr, rdotr)
        if torch.all(~not_converged):
            return x
    return x


def conjugate_gradient_solve(f_Ax: Callable[[T], T], b: T, **kwargs) -> T:
    flatten, unflatten = make_flatten_unflatten(b)

    def f_Ax_flat(v: torch.Tensor) -> torch.Tensor:
        v_unflattened: T = unflatten(v)
        result_unflattened = f_Ax(v_unflattened)
        return flatten(result_unflattened)

    return unflatten(_flat_conjugate_gradient_solve(f_Ax_flat, flatten(b), **kwargs))


def make_empirical_fisher_vp(
    batched_func_log_prob: Callable[Concatenate[ParamDict, Point[T], P], torch.Tensor],
    log_prob_params: ParamDict,
    data: Point[T],
    *args: P.args,
    **kwargs: P.kwargs,
) -> Callable[[ParamDict], ParamDict]:
    N = data[next(iter(data))].shape[0]  # type: ignore
    mean_vector = 1 / N * torch.ones(N)

    def bound_batched_func_log_prob(params: ParamDict) -> torch.Tensor:
        return batched_func_log_prob(params, data, *args, **kwargs)

    def _empirical_fisher_vp(v: ParamDict) -> ParamDict:
        def jvp_fn(log_prob_params: ParamDict) -> torch.Tensor:
            return torch.func.jvp(
                bound_batched_func_log_prob, (log_prob_params,), (v,)
            )[1]

        # Perlmutter's trick
        vjp_fn = torch.func.vjp(jvp_fn, log_prob_params)[1]
        return vjp_fn(-1 * mean_vector)[0]  # Fisher = -E[Hessian]

    return _empirical_fisher_vp


def linearize(
    model: Callable[P, Any],
    guide: Callable[P, Any],
    *,
    num_samples_outer: int,
    num_samples_inner: Optional[int] = None,
    max_plate_nesting: Optional[int] = None,
    cg_iters: Optional[int] = None,
    residual_tol: float = 1e-10,
    pointwise_influence: bool = True,
) -> Callable[Concatenate[Point[T], P], ParamDict]:
    assert isinstance(model, torch.nn.Module)
    assert isinstance(guide, torch.nn.Module)
    if num_samples_inner is None:
        num_samples_inner = num_samples_outer**2

    predictive = pyro.infer.Predictive(
        model,
        guide=guide,
        num_samples=num_samples_outer,
        parallel=True,
    )

    batched_log_prob = BatchedNMCLogPredictiveLikelihood(
        model, guide, num_samples=num_samples_inner, max_plate_nesting=max_plate_nesting
    )
    log_prob_params, batched_func_log_prob = make_functional_call(batched_log_prob)
    log_prob_params_numel: int = sum(p.numel() for p in log_prob_params.values())
    if cg_iters is None:
        cg_iters = log_prob_params_numel
    else:
        cg_iters = min(cg_iters, log_prob_params_numel)
    cg_solver = functools.partial(
        conjugate_gradient_solve, cg_iters=cg_iters, residual_tol=residual_tol
    )

    def _fn(
        points: Point[T],
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> ParamDict:
        with torch.no_grad():
            data: Point[T] = predictive(*args, **kwargs)
            data = {k: data[k] for k in points.keys()}
        fvp = make_empirical_fisher_vp(
            batched_func_log_prob, log_prob_params, data, *args, **kwargs
        )
        pinned_fvp = reset_rng_state(pyro.util.get_rng_state())(fvp)
        pinned_fvp_batched = torch.func.vmap(
            lambda v: pinned_fvp(v), randomness="different"
        )

        def bound_batched_func_log_prob(p: ParamDict) -> torch.Tensor:
            return batched_func_log_prob(p, points, *args, **kwargs)

        if pointwise_influence:
            score_fn = torch.func.jacrev(bound_batched_func_log_prob)
            point_scores = score_fn(log_prob_params)
        else:
            score_fn = torch.func.vjp(bound_batched_func_log_prob, log_prob_params)[1]
            N_pts = points[next(iter(points))].shape[0]  # type: ignore
            point_scores = score_fn(1 / N_pts * torch.ones(N_pts))[0]
            point_scores = {k: v.unsqueeze(0) for k, v in point_scores.items()}
        return cg_solver(pinned_fvp_batched, point_scores)

    return _fn
