from typing import ParamSpec, Callable, TypeVar, Optional
import torch
from pyro.infer import Predictive
from pyro.infer import Trace_ELBO
from pyro.infer.elbo import ELBOModule
from pyro.infer.importance import vectorized_importance_weights
from pyro.poutine import mask, replay, trace

P = ParamSpec("P")
Q = ParamSpec("Q")
S = TypeVar("S")
T = TypeVar("T")

Point = dict[str, T]
Guide = Callable[P, Optional[T | Point[T]]]


# guide should hide obs_names sites


def vectorized_variational_log_prob(
    model: Callable[P, T], guide: Guide[P, T], X: Point, *args, **kwargs
):
    guide_trace = trace(guide).get_trace(*args, **kwargs)
    model_trace = trace(replay(model, guide_trace)).get_trace(*args, **kwargs)
    log_probs = dict()
    for site_name, site_val in X.items():
        site = model_trace.nodes[site_name]
        assert site["type"] == "sample"
        log_probs[site_name] = site["fn"].log_prob(site_val)
    return log_probs


class LogProbModule:
    def __init__(
        self,
        model: Callable[P, T],
        guide: Guide[P, T],
        elbo: ELBOModule = Trace_ELBO,
        theta_names_to_mask: Optional[list[str]] = None,
    ):
        self.theta_names_to_mask = theta_names_to_mask
        self.model = model
        self.guide = guide
        self._log_prob_from_elbo = elbo()(mask(model, ...), mask(guide, ...))

    def log_prob(self, X: Point, *args, **kwargs) -> torch.Tensor:
        elbos = []
        for x in X:
            elbos.append(self._log_prob_from_elbo(X, *args, **kwargs))
        return torch.stack(elbos)

    def log_prob_gradient(self, X: Point, *args, **kwargs) -> torch.Tensor:
        return torch.functional.autograd(
            partial(self.log_prob(*args, **kwargs)), X, elbo.parameters()
        )


class ReparametrizableLogProb(LogProbModule):
    def log_prob(self, X: Point, *args, **kwargs) -> torch.Tensor:
        pass


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
        return pyro.sample("y", dist.Normal(y, 1))


# Create guide
guide_normal = pyro.infer.autoguide.AutoNormal(model)


def fixed_guide(x: torch.Tensor) -> None:
    pyro.sample("a", dist.Delta(torch.tensor(1.0)))
    pyro.sample("b", dist.Delta(torch.tensor(1.0)))


# Create predictive
predictive = Predictive(model, guide=fixed_guide, num_samples=1000)

samps = predictive(torch.tensor([1.0]))

# Create elbo loss
elbo = pyro.infer.Trace_ELBO(num_particles=100)(model, guide=guide_normal)


torch.autograd(elbo(torch.tensor([1.0])), elbo.parameters())

torch.autograd.functional.jacobian(
    elbo,
    torch.tensor([1.0, 2.0]),
    dict(elbo.named_parameters())["guide.locs.a_unconstrained"],
)

x0 = torch.tensor([1.0, 2.0], requires_grad=False)

elbo(x0)

x1 = torch.tensor([[1.0, 2.0], [2.0, 3.0]], requires_grad=True)


vectorized_importance_weights(
    model, guide_normal, x=x0, max_plate_nesting=4, num_samples=10000
)[0].mean()


torch.stack([torch.zeros(3), torch.zeros(3)])


elbo.parameters()
