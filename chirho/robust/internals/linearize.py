import functools
from typing import Any, Callable, Optional, TypeVar

import pyro
import torch
from typing_extensions import Concatenate, ParamSpec

from chirho.robust.internals.predictive import (
    NMCLogPredictiveLikelihood,
    PointLogPredictiveLikelihood,
)
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
    residual_tol: float = 1e-10,
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
    if cg_iters is None:
        cg_iters = b.numel()

    p = b.clone()
    r = b.clone()
    x = torch.zeros_like(b)
    z = f_Ax(p)
    rdotr = torch.dot(r, r)
    v = rdotr / torch.dot(p, z)
    newrdotr = rdotr
    mu = newrdotr / rdotr

    zeros_xr = torch.zeros_like(x)

    for _ in range(cg_iters):
        not_converged = rdotr > residual_tol
        z = torch.where(not_converged, f_Ax(p), z)
        v = torch.where(not_converged, rdotr / torch.dot(p, z), v)
        x += torch.where(not_converged, v * p, zeros_xr)
        r -= torch.where(not_converged, v * z, zeros_xr)
        newrdotr = torch.where(not_converged, torch.dot(r, r), newrdotr)
        mu = torch.where(not_converged, newrdotr / rdotr, mu)
        p = torch.where(not_converged, r + mu * p, p)
        rdotr = torch.where(not_converged, newrdotr, rdotr)

    return x


def conjugate_gradient_solve(f_Ax: Callable[[T], T], b: T, **kwargs) -> T:
    flatten, unflatten = make_flatten_unflatten(b)

    def f_Ax_flat(v: torch.Tensor) -> torch.Tensor:
        v_unflattened: T = unflatten(v)
        result_unflattened = f_Ax(v_unflattened)
        return flatten(result_unflattened)

    return unflatten(_flat_conjugate_gradient_solve(f_Ax_flat, flatten(b), **kwargs))


def make_empirical_fisher_vp(
    func_log_prob: Callable[Concatenate[ParamDict, Point[T], P], torch.Tensor],
    log_prob_params: ParamDict,
    data: Point[T],
    is_batched: bool = False,
    *args: P.args,
    **kwargs: P.kwargs,
) -> Callable[[ParamDict], ParamDict]:
    if not is_batched:
        batched_func_log_prob: Callable[
            [ParamDict, Point[T]], torch.Tensor
        ] = torch.vmap(
            lambda p, data: func_log_prob(p, data, *args, **kwargs),
            in_dims=(None, 0),
            randomness="different",
        )
    else:
        batched_func_log_prob = functools.partial(func_log_prob, *args, **kwargs)

    N = data[next(iter(data))].shape[0]  # type: ignore
    mean_vector = 1 / N * torch.ones(N)

    def bound_batched_func_log_prob(params: ParamDict) -> torch.Tensor:
        return batched_func_log_prob(params, data)

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
    is_point_estimate: bool = False,
    num_samples_inner: Optional[int] = None,
    max_plate_nesting: Optional[int] = None,
    cg_iters: Optional[int] = None,
    residual_tol: float = 1e-10,
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
    predictive_params, func_predictive = make_functional_call(predictive)

    if is_point_estimate:
        log_prob = PointLogPredictiveLikelihood(
            model,
            guide,
            max_plate_nesting=max_plate_nesting,
        )
        make_efvp = functools.partial(make_empirical_fisher_vp, is_batched=True)
    else:
        log_prob = NMCLogPredictiveLikelihood(
            model,
            guide,
            num_samples=num_samples_inner,
            max_plate_nesting=max_plate_nesting,
        )
        make_efvp = functools.partial(make_empirical_fisher_vp, is_batched=False)

    log_prob_params, func_log_prob = make_functional_call(log_prob)
    log_prob_params_numel: int = sum(p.numel() for p in log_prob_params.values())
    if cg_iters is None:
        cg_iters = log_prob_params_numel
    else:
        cg_iters = min(cg_iters, log_prob_params_numel)
    cg_solver = functools.partial(
        conjugate_gradient_solve, cg_iters=cg_iters, residual_tol=residual_tol
    )

    def _fn(points: Point[T], *args: P.args, **kwargs: P.kwargs) -> ParamDict:
        with torch.no_grad():
            data: Point[T] = func_predictive(predictive_params, *args, **kwargs)
            data = {k: data[k] for k in points.keys()}
        fvp = make_efvp(func_log_prob, log_prob_params, data, *args, **kwargs)
        pinned_fvp = reset_rng_state(pyro.util.get_rng_state())(fvp)
        if not is_point_estimate:
            batched_func_log_prob = torch.vmap(
                lambda p, data: func_log_prob(p, data, *args, **kwargs),
                in_dims=(None, 0),
                randomness="different",
            )
        else:
            batched_func_log_prob = functools.partial(func_log_prob, *args, **kwargs)
        if log_prob_params_numel > points[next(iter(points))].shape[0]:
            score_fn = torch.func.jacrev(batched_func_log_prob)
        else:
            score_fn = torch.func.jacfwd(batched_func_log_prob, randomness="different")
        point_scores: ParamDict = score_fn(log_prob_params, points)
        return torch.func.vmap(
            lambda v: cg_solver(pinned_fvp, v), randomness="different"
        )(point_scores)

    return _fn
