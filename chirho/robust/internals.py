import math
import collections
import pyro
from typing import Container, ParamSpec, Callable, Tuple, TypeVar, Optional, Dict, List, Protocol
import torch
from pyro.infer import Predictive
from pyro.infer import Trace_ELBO
from pyro.infer.elbo import ELBOModule
from pyro.infer.importance import vectorized_importance_weights
from pyro.poutine import block, replay, trace, mask
from chirho.indexed.handlers import DependentMaskMessenger
from chirho.observational.handlers import condition
from torch.func import functional_call
from functools import partial
from utils import conjugate_gradient_solve

pyro.settings.set(module_local_params=True)

P = ParamSpec("P")
Q = ParamSpec("Q")
S = TypeVar("S")
T = TypeVar("T") # This will be a torch.Tensor usually

Point = dict[str, T]
Guide = Callable[P, Optional[T | Point[T]]]


def make_empirical_inverse_fisher_vp(
    log_prob: torch.nn.Module,
    **solver_kwargs,
) -> Callable:

    fvp = make_empirical_fisher_vp(log_prob)
    return lambda v: conjugate_gradient_solve(fvp, v, **solver_kwargs)


def make_empirical_fisher_vp(
    log_prob: torch.nn.Module,
) -> Callable:

    def _empirical_fisher_vp(v: T) -> T:
        params = dict(log_prob.named_parameters())
        vnew = torch.func.jvp(partial(torch.func.functional_call, log_prob), params, v)
        (_, vjp_fn) = torch.func.vjp(partial(torch.func.functional_call, log_prob), params)
        return vjp_fn(vnew)

    return _empirical_fisher_vp


class UnmaskNamedSites(DependentMaskMessenger):
    names: Container[str]

    def __init__(self, names: Container[str]):
        self.names = names

    def get_mask(
        self, dist: pyro.distributions.Distribution, value: Optional[torch.Tensor], device: torch.device, name: str
    ) -> torch.Tensor:
        return torch.tensor(name in self.names, device=device)


class NMCLogLikelihood(torch.nn.Module):

    def __init__(
        self,
        model: pyro.nn.PyroModule,
        guide: pyro.nn.PyroModule,
        num_samples: int,
    ):
        super().__init__()
        self.model = model
        self.guide = guide
        self.num_samples = num_samples

    def forward(self, data: Point[torch.Tensor], *args, **kwargs) -> torch.Tensor:
        num_monte_carlo_outer = data[next(iter(data))].shape[0]
        # if num_monte_inner is None:
        #     # Optimal scaling for inner expectation: 
        #     # see https://arxiv.org/pdf/1709.06181.pdf
        #     num_monte_inner = num_monte_carlo_outer ** 2    
        
        log_weights = []
        for i in range(num_monte_carlo_outer):
            log_weights_i = []
            for j in range(self.num_samples):
                masked_guide = pyro.poutine.mask(mask=False)(self.guide) # Mask all sites in guide
                masked_model = UnmaskNamedSites(names=set(data.keys()))(condition(data={k: v[i] for k, v in data.items()})(self.model))
                log_weight_ij = pyro.infer.Trace_ELBO().differentiable_loss(masked_model, masked_guide, *args, **kwargs)
                log_weights_i.append(log_weight_ij)
            log_weight_i = torch.logsumexp(torch.stack(log_weights_i), dim=0) - math.log(self.num_samples)
            log_weights.append(log_weight_i)

        log_weights = torch.stack(log_weights)
        assert log_weights.shape == (num_monte_carlo_outer,)
        return log_weights / (num_monte_carlo_outer ** 0.5)


class NMCLogLikelihoodSingle(NMCLogLikelihood):
    def forward(self, data: Point[torch.Tensor], *args, **kwargs) -> torch.Tensor:
        masked_guide = pyro.poutine.mask(mask=False)(self.guide) # Mask all sites in guide
        masked_model = UnmaskNamedSites(names=set(data.keys()))(condition(data=data)(self.model))
        log_weights = pyro.infer.importance.vectorized_importance_weights(masked_model, masked_guide, *args, num_samples=self.num_samples, max_plate_nesting=1, **kwargs)[0]
        return torch.logsumexp(log_weights * self.guide.zzz.w, dim=0) - math.log(self.num_samples)


class DummyAutoNormal(pyro.infer.autoguide.AutoNormal):

    def __getattr__(self, name):
        # PyroParams trigger pyro.param statements.
        if "_pyro_params" in self.__dict__:
            _pyro_params = self.__dict__["_pyro_params"]
            if name in _pyro_params:
                constraint, event_dim = _pyro_params[name]
                unconstrained_value = getattr(self, name + "_unconstrained")
                import weakref
                constrained_value = torch.distributions.transform_to(constraint)(unconstrained_value)
                constrained_value.unconstrained = weakref.ref(unconstrained_value)
                return constrained_value
        return super().__getattr__(name)


if __name__ == "__main__":
    import pyro
    import pyro.distributions as dist

    # Create simple pyro model
    class SimpleModel(pyro.nn.PyroModule):
        def forward(self):
            a = pyro.sample("a", dist.Normal(0, 1))
            b = pyro.sample("b", dist.Normal(0, 1))
            return pyro.sample("y", dist.Normal(a + b, 1))

    model = SimpleModel()

    # Create guide on latents a and b
    num_monte_carlo_outer = 100
    guide = DummyAutoNormal(block(model, hide=["y"]))
    zzz = pyro.nn.PyroModule()
    zzz.w = pyro.nn.PyroParam(torch.rand(10), dist.constraints.positive)
    guide()
    guide.zzz = zzz
    print(dict(guide.named_parameters()))
    data = Predictive(model, guide=guide, num_samples=num_monte_carlo_outer, return_sites=["y"])()

    # Create log likelihood function
    log_prob = NMCLogLikelihoodSingle(model, guide, num_samples=10)

    log_prob_func = torch.func.vmap(
        torch.func.functionalize(pyro.validation_enabled(False)(partial(torch.func.functional_call, log_prob))),
        # pyro.validation_enabled(False)(partial(torch.func.functional_call, log_prob)),
        in_dims=(None, 0),
        randomness='different'
    )

    print(log_prob_func(dict(log_prob.named_parameters()), data)[0])

    # func
    grad_log_prob = torch.func.vjp(log_prob_func, dict(log_prob.named_parameters()), data)[1]
    print(grad_log_prob(torch.ones(num_monte_carlo_outer))[0])

    # autograd.functional
    param_dict = collections.OrderedDict(log_prob.named_parameters())
    print(dict(zip(param_dict.keys(), torch.autograd.functional.vjp(
        lambda *params: log_prob_func(dict(zip(param_dict.keys(), params)), data),
        tuple(param_dict.values()),
        torch.ones(num_monte_carlo_outer)
    )[1])))

    # print(torch.autograd.grad(partial(torch.func.functional_call, log_prob)(dict(log_prob.named_parameters()), data), tuple(log_prob.parameters())))
    # fvp = make_empirical_fisher_vp(log_prob) 

    # v = tuple(torch.ones_like(p) for p in guide.parameters())

    # print(v, fvp(v))
    




