from typing import ParamSpec, Callable, TypeVar, Optional
import torch
from pyro.infer import Predictive
from pyro.infer import Trace_ELBO
from pyro.infer.elbo import ELBOModule
from pyro.infer.importance import vectorized_importance_weights
from pyro.poutine import mask

P = ParamSpec("P")
Q = ParamSpec("Q")
S = TypeVar("S")
T = TypeVar("T")

Point = dict[str, T]
Guide = Callable[P, Optional[T | Point[T]]]


class LogProbModule:
    def __init__(
        self,
        model: Callable[P, T],
        guide: Guide[P, T],
        # elbo: ELBOModule = Trace_ELBO,
        theta_names_to_mask: Optional[list[str]] = None,
    ):
        self.theta_names_to_mask = theta_names_to_mask
        # self._log_prob_from_elbo = elbo()(mask(model, ...), mask(guide, ...))

    def log_prob(self, X, *args, **kwargs) -> torch.Tensor:
        raise NotImplementedError

    def log_prob_gradient(self, X, *args, **kwargs) -> torch.Tensor:
        raise NotImplementedError


class ReparametrizableLogProbModule(LogProbModule):
    def __init__(
        self,
        model: Callable[P, T],
        guide: Guide[P, T],
        elbo: ELBOModule = Trace_ELBO,
        theta_names_to_mask: Optional[list[str]] = None,
    ):
        self._log_prob_from_elbo = elbo()(mask(model, ...), mask(guide, ...))

    # Use vmap here to get elbo at multiple points
    def log_prob(self, X, *args, **kwargs) -> torch.Tensor:
        elbos = []
        for x in X:
            elbos.append(self._log_prob_from_elbo(X, *args, **kwargs))
        return torch.stack(elbos)

    def log_prob_gradient(self, X, *args, **kwargs) -> torch.Tensor:
        return torch.functional.autograd(
            partial(self.log_prob(*args, **kwargs)), X, elbo.parameters()
        )


# For continous latents, vectorized importance weights
# https://docs.pyro.ai/en/stable/inference_algos.html#pyro.infer.importance.vectorized_importance_weights

# Predictive(model, guide)


import pyro
import pyro.distributions as dist


# Create simple pyro model
def model(x: torch.Tensor) -> torch.Tensor:
    a = pyro.sample("a", dist.Normal(0, 1))
    b = pyro.sample("b", dist.Normal(0, 1))
    with pyro.plate("data", x.shape[0]):
        y = a * x + b
        pyro.sample("y", dist.Normal(y, 1))


# Create guide
guide_normal = pyro.infer.autoguide.AutoNormal(model)


def fixed_guide(x: torch.Tensor) -> None:
    pyro.sample("a", dist.Delta(torch.tensor(1.0)))
    pyro.sample("b", dist.Delta(torch.tensor(1.0)))


# Create predictive
predictive = Predictive(model, guide=fixed_guide, num_samples=1000)

samps = predictive(torch.tensor([1.0]))

# Create elbo loss
elbo = pyro.infer.Trace_ELBO(num_particles=10000)(model, guide=guide_normal)


torch.autograd(elbo(torch.tensor([1.0])), elbo.parameters())

torch.autograd.functional.jacobian(elbo, torch.tensor([1.0]), elbo.parameters())

x0 = torch.tensor([1.0, 2.0], requires_grad=True)

elbo(x0)

x1 = torch.tensor([[1.0, 2.0], [2.0, 3.0]], requires_grad=True)


vectorized_importance_weights(
    model, guide_normal, x=x0, max_plate_nesting=4, num_samples=10000
)[0].mean()
