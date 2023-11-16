from typing import ParamSpec, Callable, TypeVar, Optional, Dict, List
import torch
from pyro.infer import Predictive
from pyro.infer import Trace_ELBO
from pyro.infer.elbo import ELBOModule
from pyro.infer.importance import vectorized_importance_weights
from pyro.poutine import block, replay, trace, mask

P = ParamSpec("P")
Q = ParamSpec("Q")
S = TypeVar("S")
T = TypeVar("T")

Point = dict[str, T]
Guide = Callable[P, Optional[T | Point[T]]]


def _shuffle_dict(d: dict[str, T]):
    """
    Shuffle values of a dictionary in first batch dimension
    """
    return {k: v[torch.randperm(v.shape[0])] for k, v in d.items()}


# Need to add vectorize function from vectorized_importance_weights


# Issue: gradients detached in predictives
def vectorized_variational_log_prob(
    model: Callable[P, T],
    guide: Guide[P, T],
    trace_predictive: Dict,
    obs_names: List[str],
    # num_particles: int = 1, # TODO: support this next
    *args,
    **kwargs
):
    """
    See eq. 3 in http://approximateinference.org/2017/accepted/TangRanganath2017.pdf
    """
    latent_params_trace = _shuffle_dict(
        {k: v.clone() for k, v in trace_predictive.items() if k not in obs_names}
    )
    obs_vars_trace = {
        k: v.clone().detach() for k, v in trace_predictive.items() if k in obs_names
    }
    import pdb

    pdb.set_trace()
    model_trace = trace(replay(model, latent_params_trace)).get_trace(*args, **kwargs)

    N_samples = next(iter(latent_params_trace.values())).shape[0]

    log_probs = torch.zeros(N_samples)
    for site_name, site_val in obs_vars_trace.items():
        site = model_trace.nodes[site_name]
        assert site["type"] == "sample"
        log_probs += site["fn"].log_prob(site_val)
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


def stochastic_variational_log_likelihood_fn(
    flat_theta: torch.tensor, X: Dict[str, torch.Tensor]
):
    pass


# For continous latents, vectorized importance weights
# https://docs.pyro.ai/en/stable/inference_algos.html#pyro.infer.importance.vectorized_importance_weights

# Predictive(model, guide)

if __name__ == "__main__":
    import pyro
    import pyro.distributions as dist

    # Create simple pyro model
    def model():
        a = pyro.sample("a", dist.Normal(0, 1))
        b = pyro.sample("b", dist.Normal(0, 1))
        return pyro.sample("y", dist.Normal(a + b, 1))

    # Create guide on latents a and b
    guide = pyro.infer.autoguide.AutoNormal(block(model, hide=["y"]))
    # with pyro.poutine.trace() as tr:
    #     guide()
    # print(tr.trace.nodes.keys())
    # Create predictive
    predictive = Predictive(model, guide=guide, num_samples=100)
    # with pyro.poutine.trace() as tr:
    X = predictive()

    vectorized_variational_log_prob(model, guide, X, ["y"])

    # print(X)
    # import pdb

    # pdb.set_trace()
