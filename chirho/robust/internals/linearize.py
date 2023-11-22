import functools
from typing import Callable, Concatenate, Optional, ParamSpec, TypeVar

import pyro
import torch

from chirho.robust.internals.predictive import NMCLogPredictiveLikelihood
from chirho.robust.internals.utils import conjugate_gradient_solve, make_functional_call
from chirho.robust.ops import Model, ParamDict, Point

pyro.settings.set(module_local_params=True)

P = ParamSpec("P")
Q = ParamSpec("Q")
S = TypeVar("S")
T = TypeVar("T")


def make_empirical_fisher_vp(
    func_log_prob: Callable[Concatenate[ParamDict, Point[T], P], torch.Tensor],
    log_prob_params: ParamDict,
    data: Point[T],
    *args: P.args,
    **kwargs: P.kwargs,
) -> Callable[[ParamDict], ParamDict]:
    batched_func_log_prob: Callable[[ParamDict, Point[T]], torch.Tensor] = torch.vmap(
        lambda p, data: func_log_prob(p, data, *args, **kwargs),
        in_dims=(None, 0),
        randomness="different",
    )

    def bound_batched_func_log_prob(params: ParamDict) -> torch.Tensor:
        return batched_func_log_prob(params, data)

    def jvp_fn(v: ParamDict) -> torch.Tensor:
        return torch.func.jvp(bound_batched_func_log_prob, (log_prob_params,), (v,))[1]

    vjp_fn = torch.func.vjp(bound_batched_func_log_prob, log_prob_params)[1]

    def _empirical_fisher_vp(v: ParamDict) -> ParamDict:
        jvp_log_prob_v = jvp_fn(v)
        return vjp_fn(jvp_log_prob_v / jvp_log_prob_v.shape[0])[0]

    return _empirical_fisher_vp


def linearize(
    model: Model[P],
    guide: Model[P],
    *,
    num_samples_outer: int,
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

    log_prob = NMCLogPredictiveLikelihood(
        model, guide, num_samples=num_samples_inner, max_plate_nesting=max_plate_nesting
    )
    log_prob_params, func_log_prob = make_functional_call(log_prob)
    score_fn = torch.func.grad(func_log_prob)

    cg_solver = functools.partial(
        conjugate_gradient_solve, cg_iters=cg_iters, residual_tol=residual_tol
    )

    @functools.wraps(score_fn)
    def _fn(point: Point[T], *args: P.args, **kwargs: P.kwargs) -> ParamDict:
        with torch.no_grad():
            data: Point[T] = func_predictive(predictive_params, *args, **kwargs)
            data = {k: data[k] for k in point.keys()}
        fvp = make_empirical_fisher_vp(
            func_log_prob, log_prob_params, data, *args, **kwargs
        )
        point_score: ParamDict = score_fn(log_prob_params, point, *args, **kwargs)
        return cg_solver(fvp, point_score)

    return _fn
