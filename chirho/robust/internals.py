import functools
import math
from typing import (
    Callable,
    Concatenate,
    Container,
    Dict,
    Generic,
    Optional,
    ParamSpec,
    Tuple,
    TypeVar,
)

import pyro
import torch

from chirho.indexed.handlers import DependentMaskMessenger
from chirho.observational.handlers import condition
from chirho.robust.ops import Model, ParamDict, Point

pyro.settings.set(module_local_params=True)

P = ParamSpec("P")
Q = ParamSpec("Q")
S = TypeVar("S")
T = TypeVar("T")


@functools.singledispatch
def make_flatten_unflatten(
    v,
) -> Tuple[Callable[[T], torch.Tensor], Callable[[torch.Tensor], T]]:
    raise NotImplementedError


@make_flatten_unflatten.register(dict)
def _make_flatten_unflatten_dict(d: Dict[str, torch.Tensor]):
    def flatten(d: Dict[str, torch.Tensor]) -> torch.Tensor:
        r"""
        Flatten a dictionary of tensors into a single vector.
        """
        return torch.cat([v.flatten() for k, v in d.items()])

    def unflatten(x: torch.Tensor) -> Dict[str, torch.Tensor]:
        r"""
        Unflatten a vector into a dictionary of tensors.
        """
        return dict(
            zip(
                d.keys(),
                [
                    v_flat.reshape(v.shape)
                    for v, v_flat in zip(
                        d.values(), torch.split(x, [v.numel() for k, v in d.items()])
                    )
                ],
            )
        )

    return flatten, unflatten


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

    Notes: This code is adapted from https://github.com/rlworkgroup/garage/blob/master/src/garage/torch/optimizers/conjugate_gradient_optimizer.py
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

        # rdotr = newrdotr
        # if rdotr < residual_tol:
        #     break
    return x


def conjugate_gradient_solve(f_Ax: Callable[[T], T], b: T, **kwargs) -> T:
    flatten, unflatten = make_flatten_unflatten(b)

    def f_Ax_flat(v: torch.Tensor) -> torch.Tensor:
        v_unflattened: T = unflatten(v)
        result_unflattened = f_Ax(v_unflattened)
        return flatten(result_unflattened)

    return unflatten(_flat_conjugate_gradient_solve(f_Ax_flat, flatten(b), **kwargs))


def make_functional_call(
    mod: Callable[P, T]
) -> Tuple[ParamDict, Callable[Concatenate[ParamDict, P], T]]:
    assert isinstance(mod, torch.nn.Module)
    param_dict: ParamDict = dict(mod.named_parameters())

    @torch.func.functionalize
    def mod_func(params: ParamDict, *args: P.args, **kwargs: P.kwargs) -> T:
        with pyro.validation_enabled(False):
            return torch.func.functional_call(mod, params, args, dict(**kwargs))

    return param_dict, mod_func


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

    jvp_fn = functools.partial(
        torch.func.jvp, bound_batched_func_log_prob, (log_prob_params,)
    )
    vjp_fn = torch.func.vjp(bound_batched_func_log_prob, log_prob_params)[1]

    def _empirical_fisher_vp(v: ParamDict) -> ParamDict:
        jvp_log_prob_v = jvp_fn((v,))[1]
        return vjp_fn(jvp_log_prob_v / jvp_log_prob_v.shape[0])[0]

    return _empirical_fisher_vp


class UnmaskNamedSites(DependentMaskMessenger):
    names: Container[str]

    def __init__(self, names: Container[str]):
        self.names = names

    def get_mask(
        self,
        dist: pyro.distributions.Distribution,
        value: Optional[torch.Tensor],
        device: torch.device = torch.device("cpu"),
        name: Optional[str] = None,
    ) -> torch.Tensor:
        return torch.tensor(name is None or name in self.names, device=device)


class NMCLogPredictiveLikelihood(Generic[P, T], torch.nn.Module):
    def __init__(
        self,
        model: torch.nn.Module,
        guide: torch.nn.Module,
        *,
        num_samples: int = 1,
        max_plate_nesting: int = 1,
    ):
        super().__init__()
        self.model = model
        self.guide = guide
        self.num_samples = num_samples
        self.max_plate_nesting = max_plate_nesting

    def forward(
        self, data: Point[T], *args: P.args, **kwargs: P.kwargs
    ) -> torch.Tensor:
        masked_guide = pyro.poutine.mask(mask=False)(self.guide)
        masked_model = UnmaskNamedSites(names=set(data.keys()))(
            condition(data=data)(self.model)
        )
        log_weights = pyro.infer.importance.vectorized_importance_weights(
            masked_model,
            masked_guide,
            *args,
            num_samples=self.num_samples,
            max_plate_nesting=self.max_plate_nesting,
            **kwargs,
        )[0]
        return torch.logsumexp(log_weights, dim=0) - math.log(self.num_samples)


def linearize(
    model: Model[P],
    guide: Model[P],
    *,
    max_plate_nesting: int,
    num_samples_outer: int,
    num_samples_inner: Optional[int] = None,
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
