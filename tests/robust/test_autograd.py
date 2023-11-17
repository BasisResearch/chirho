import math
import collections
import functools
import pyro
from typing import Concatenate, Container, ParamSpec, Callable, Tuple, TypeVar, Optional, Mapping, Dict, List, Protocol
import torch
import pyro.distributions as dist
from pyro.infer import Predictive
from pyro.infer import Trace_ELBO
from pyro.infer.elbo import ELBOModule
from pyro.infer.importance import vectorized_importance_weights
from pyro.poutine import block, replay, trace, mask
from chirho.indexed.handlers import DependentMaskMessenger
from chirho.observational.handlers import condition
from torch.func import functional_call
from functools import partial

pyro.settings.set(module_local_params=True)

P = ParamSpec("P")
Q = ParamSpec("Q")
S = TypeVar("S")
T = TypeVar("T") # This will be a torch.Tensor usually

Point = dict[str, T]
Guide = Callable[P, Optional[T | Point[T]]]


@functools.singledispatch
def make_flatten_unflatten(v) -> Tuple[Callable[[T], torch.Tensor], Callable[[torch.Tensor], T]]:
    raise NotImplementedError


@make_flatten_unflatten.register(tuple)
def _make_flatten_unflatten_tuple(v: Tuple[torch.Tensor, ...]):
    sizes = [x.size() for x in v]

    def flatten(xs: Tuple[torch.Tensor, ...]) -> torch.Tensor:
        return torch.cat([x.reshape(-1) for x in xs], dim=0)

    def unflatten(x: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        tensors = []
        i = 0
        for size in sizes:
            num_elements = torch.prod(torch.tensor(size))
            tensors.append(x[i : i + num_elements].view(size))
            i += num_elements
        return tuple(tensors)

    return flatten, unflatten


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
    f_Ax: Callable[[torch.Tensor], torch.Tensor], b: torch.Tensor, *, cg_iters: Optional[int] = None, residual_tol: float = 1e-10
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

    Notes: This code is copied from https://github.com/rlworkgroup/garage/blob/master/src/garage/torch/optimizers/conjugate_gradient_optimizer.py
    """
    if cg_iters is None:
        cg_iters = b.numel()

    p = b.clone()
    r = b.clone()
    x = torch.zeros_like(b)
    rdotr = torch.dot(r, r)

    import pdb; pdb.set_trace()

    for _ in range(cg_iters):
        z = f_Ax(p)
        import pdb; pdb.set_trace()
        v = rdotr / torch.dot(p, z)
        x += v * p
        r -= v * z
        newrdotr = torch.dot(r, r)
        mu = newrdotr / rdotr
        p = r + mu * p
        import pdb; pdb.set_trace()

        # Still executes loop but effectively stops update (can't break loop since we're using vmap)
        # rdotr = torch.where(rdotr < residual_tol, rdotr, newrdotr)
        # rdotr = newrdotr
        # if rdotr < residual_tol:
        #     break
    import pdb; pdb.set_trace()
    return x


def conjugate_gradient_solve(f_Ax: Callable[[T], T], b: T, **kwargs) -> T:
    flatten, unflatten = make_flatten_unflatten(b)

    def f_Ax_flat(v: torch.Tensor) -> torch.Tensor:
        v_unflattened = unflatten(v)
        result_unflattened = f_Ax(v_unflattened)
        return flatten(result_unflattened)

    return unflatten(_flat_conjugate_gradient_solve(f_Ax_flat, flatten(b), **kwargs))


def make_functional_call(
    mod: Callable[P, T]
) -> Tuple[
    Mapping[str, torch.Tensor],
    Callable[Concatenate[Mapping[str, torch.Tensor], P], T]
]:
    assert isinstance(mod, torch.nn.Module)
    return dict(mod.named_parameters()), torch.func.functionalize(pyro.validation_enabled(False)(functools.partial(torch.func.functional_call, mod)))


def make_bound_batched_func_log_prob(
    log_prob: Callable[[T], torch.Tensor],
    data: T
) -> Tuple[Mapping[str, torch.Tensor], Callable[[Mapping[str, torch.Tensor]], torch.Tensor]]:

    assert isinstance(log_prob, torch.nn.Module)
    log_prob_params_and_fn = make_functional_call(log_prob)
    log_prob_params: Mapping[str, torch.Tensor] = log_prob_params_and_fn[0]
    func_log_prob: Callable[[Mapping[str, torch.Tensor], T], torch.Tensor] = log_prob_params_and_fn[1]

    batched_func_log_prob: Callable[[Mapping[str, torch.Tensor], T], torch.Tensor] = torch.vmap(
        func_log_prob,
        in_dims=(None, 0),
        randomness='different'
    )

    def bound_batched_func_log_prob(params: Mapping[str, torch.Tensor]) -> torch.Tensor:
        return batched_func_log_prob(params, data)

    return log_prob_params, bound_batched_func_log_prob


def make_empirical_fisher_vp(
    log_prob: Callable[[T], torch.Tensor],
    data: T
) -> Callable[[Mapping[str, torch.Tensor]], Mapping[str, torch.Tensor]]:

    log_prob_params, bound_batched_func_log_prob = make_bound_batched_func_log_prob(log_prob, data)

    def _empirical_fisher_vp(v: Mapping[str, torch.Tensor]) -> Mapping[str, torch.Tensor]:
        (_, vnew) = torch.func.jvp(bound_batched_func_log_prob, (log_prob_params,), (v,))
        (_, vjp_fn) = torch.func.vjp(bound_batched_func_log_prob, log_prob_params)
        result: Mapping[str, torch.Tensor] = vjp_fn(vnew / vnew.shape[0])[0]
        import pdb; pdb.set_trace()
        # result is batched over datapoints (via vmap), so we must sum out the batch dimension 0?
        # return {k: torch.sum(v, dim=0) for k, v in result.items()}
        assert result.keys() == v.keys() and all(result[k].shape == v[k].shape for k in result.keys())
        return result

    return _empirical_fisher_vp


def make_empirical_inverse_fisher_vp(
    log_prob: Callable[[T], torch.Tensor],
    data: T,
    **solver_kwargs,
) -> Callable[[Mapping[str, torch.Tensor]], Mapping[str, torch.Tensor]]:

    assert isinstance(log_prob, torch.nn.Module)
    fvp = make_empirical_fisher_vp(log_prob, data)
    return functools.partial(conjugate_gradient_solve, fvp, **solver_kwargs)


class UnmaskNamedSites(DependentMaskMessenger):
    names: Container[str]

    def __init__(self, names: Container[str]):
        self.names = names

    def get_mask(
        self, dist: pyro.distributions.Distribution, value: Optional[torch.Tensor], device: torch.device, name: str
    ) -> torch.Tensor:
        return torch.tensor(name in self.names, device=device)


class NMCLogPredictiveLikelihood(torch.nn.Module):

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

    def forward(self, data: Point[torch.Tensor], *args, **kwargs) -> torch.Tensor:
        masked_guide = pyro.poutine.mask(mask=False)(self.guide)
        masked_model = UnmaskNamedSites(names=set(data.keys()))(condition(data=data)(self.model))
        log_weights = pyro.infer.importance.vectorized_importance_weights(
            masked_model,
            masked_guide,
            *args,
            num_samples=self.num_samples,
            max_plate_nesting=self.max_plate_nesting,
            **kwargs
        )[0]
        return torch.logsumexp(log_weights, dim=0) - math.log(self.num_samples)


def test_nmc_log_likelihood():

    # Create simple pyro model
    class SimpleModel(pyro.nn.PyroModule):
        def forward(self):
            a = pyro.sample("a", dist.Normal(0, 1))
            b = pyro.sample("b", dist.Normal(0, 1))
            return pyro.sample("y", dist.Normal(a + b, 1))

    class SimpleGuide(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.loc_a = torch.nn.Parameter(torch.rand(()))
            self.loc_b = torch.nn.Parameter(torch.rand(()))

        def forward(self):
            a = pyro.sample("a", dist.Normal(self.loc_a, 1.))
            b = pyro.sample("b", dist.Normal(self.loc_b, 1.))

    model = SimpleModel()
    guide = SimpleGuide()

    # Create guide on latents a and b
    num_samples_outer = 100
    data = pyro.infer.Predictive(model, guide=guide, num_samples=num_samples_outer, return_sites=["y"], parallel=True)()

    # Create log likelihood function
    log_prob = NMCLogPredictiveLikelihood(model, guide, num_samples=10000, max_plate_nesting=1)

    v = {k: torch.ones_like(v) for k, v in log_prob.named_parameters()}

    # fvp = make_empirical_fisher_vp(log_prob, data) 
    # print(v, fvp(v))

    flatten_v, unflatten_v = make_flatten_unflatten(v)
    assert unflatten_v(flatten_v(v)) == v
    fivp = make_empirical_inverse_fisher_vp(log_prob, data, cg_iters = 1)
    print(v, fivp(v))
