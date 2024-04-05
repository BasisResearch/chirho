import math
from typing import Callable, Optional, Tuple, TypedDict, TypeVar

import pyro
import pyro.distributions as dist
import torch
from pyro.nn import PyroModule

from chirho.observational.handlers import condition
from chirho.robust.internals.utils import ParamDict
from chirho.robust.ops import Point

pyro.settings.set(module_local_params=True)
T = TypeVar("T")


class SimpleModel(PyroModule):
    def __init__(
        self,
        link_fn: Callable[..., dist.Distribution] = lambda mu: dist.Normal(mu, 1.0),
    ):
        super().__init__()
        self.link_fn = link_fn

    def forward(self):
        a = pyro.sample("a", dist.Normal(0, 1))
        with pyro.plate("data", 3, dim=-1):
            b = pyro.sample("b", dist.Normal(a, 1))
            return pyro.sample("y", dist.Normal(b, 1))


class SimpleGuide(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.loc_a = torch.nn.Parameter(torch.rand(()))
        self.loc_b = torch.nn.Parameter(torch.rand((3,)))

    def forward(self):
        a = pyro.sample("a", dist.Normal(self.loc_a, 1))
        with pyro.plate("data", 3, dim=-1):
            b = pyro.sample("b", dist.Normal(self.loc_b, 1))
            return {"a": a, "b": b}


class GaussianModel(PyroModule):
    def __init__(self, cov_mat: torch.Tensor):
        super().__init__()
        self.register_buffer("cov_mat", cov_mat)

    def forward(self, loc):
        pyro.sample(
            "x", dist.MultivariateNormal(loc=loc, covariance_matrix=self.cov_mat)
        )


# Note: `gaussian_log_prob` is separate from the GaussianModel above because of upstream obstacles
# in the interaction between `pyro.nn.PyroModule` and `torch.func`.
# See https://github.com/BasisResearch/chirho/issues/393
def gaussian_log_prob(params: ParamDict, data_point: Point[T], cov_mat) -> T:
    with pyro.validation_enabled(False):
        return dist.MultivariateNormal(
            loc=params["loc"], covariance_matrix=cov_mat
        ).log_prob(data_point["x"])


class DataConditionedModel(PyroModule):
    r"""
    Helper class for conditioning on data.
    """

    def __init__(self, model: PyroModule):
        super().__init__()
        self.model = model

    def forward(self, D: Point[torch.Tensor]):
        with condition(data=D):
            # Assume first dimension corresponds to # of datapoints
            N = D[next(iter(D))].shape[0]
            return self.model.forward(N=N)


class HighDimLinearModel(pyro.nn.PyroModule):
    def __init__(
        self,
        p: int,
        link_fn: Callable[..., dist.Distribution] = lambda mu: dist.Normal(mu, 1.0),
        prior_scale: Optional[float] = None,
    ):
        super().__init__()
        self.p = p
        self.link_fn = link_fn
        if prior_scale is None:
            self.prior_scale = 1 / math.sqrt(self.p)
        else:
            self.prior_scale = prior_scale

    def sample_outcome_weights(self):
        return pyro.sample(
            "outcome_weights",
            dist.Normal(0.0, self.prior_scale).expand((self.p,)).to_event(1),
        )

    def sample_intercept(self):
        return pyro.sample("intercept", dist.Normal(0.0, 1.0))

    def sample_propensity_weights(self):
        return pyro.sample(
            "propensity_weights",
            dist.Normal(0.0, self.prior_scale).expand((self.p,)).to_event(1),
        )

    def sample_treatment_weight(self):
        return pyro.sample("treatment_weight", dist.Normal(0.0, 1.0))

    def sample_covariate_loc_scale(self):
        loc = pyro.sample(
            "covariate_loc", dist.Normal(0.0, 1.0).expand((self.p,)).to_event(1)
        )
        scale = pyro.sample(
            "covariate_scale", dist.LogNormal(0, 1).expand((self.p,)).to_event(1)
        )
        return loc, scale

    def forward(self, N: int = 1):
        intercept = self.sample_intercept()
        outcome_weights = self.sample_outcome_weights()
        propensity_weights = self.sample_propensity_weights()
        tau = self.sample_treatment_weight()
        x_loc, x_scale = self.sample_covariate_loc_scale()
        with pyro.plate("obs", N, dim=-1):
            X = pyro.sample("X", dist.Normal(x_loc, x_scale).to_event(1))
            A = pyro.sample(
                "A",
                dist.Bernoulli(
                    logits=torch.einsum("...np,...p->...n", X, propensity_weights)
                ),
            )
            return pyro.sample(
                "Y",
                self.link_fn(
                    torch.einsum("...np,...p->...n", X, outcome_weights)
                    + A * tau
                    + intercept
                ),
            )


class KnownCovariateDistModel(HighDimLinearModel):
    def sample_covariate_loc_scale(self):
        return torch.zeros(self.p), torch.ones(self.p)


class BenchmarkLinearModel(HighDimLinearModel):
    def __init__(
        self,
        p: int,
        link_fn: Callable[..., dist.Distribution],
        alpha: int,
        beta: int,
        treatment_weight: float = 0.0,
    ):
        super().__init__(p, link_fn)
        self.alpha = alpha  # sparsity of propensity weights
        self.beta = beta  # sparisty of outcome weights
        self.treatment_weight = treatment_weight

    def sample_outcome_weights(self):
        outcome_weights = 1 / math.sqrt(self.beta) * torch.ones(self.p)
        outcome_weights[self.beta :] = 0.0
        return outcome_weights

    def sample_treatment_null_weight(self):
        return torch.tensor(0.0)

    def sample_propensity_weights(self):
        propensity_weights = 1 / math.sqrt(self.alpha) * torch.ones(self.p)
        propensity_weights[self.alpha :] = 0.0
        return propensity_weights

    def sample_treatment_weight(self):
        return torch.tensor(self.treatment_weight)

    def sample_intercept(self):
        return torch.tensor(0.0)

    def sample_covariate_loc_scale(self):
        return torch.zeros(self.p), torch.ones(self.p)


class MLEGuide(torch.nn.Module):
    def __init__(self, mle_est: ParamDict):
        super().__init__()
        self.names = list(mle_est.keys())
        for name, value in mle_est.items():
            setattr(self, name + "_param", torch.nn.Parameter(value))

    def forward(self, *args, **kwargs):
        for name in self.names:
            value = getattr(self, name + "_param")
            pyro.sample(name, dist.Delta(value))


class ATETestPoint(TypedDict):
    X: torch.Tensor
    A: torch.Tensor
    Y: torch.Tensor


class ATEParamDict(TypedDict):
    propensity_weights: torch.Tensor
    outcome_weights: torch.Tensor
    treatment_weight: torch.Tensor
    intercept: torch.Tensor


def closed_form_ate_correction(
    X_test: ATETestPoint, theta: ATEParamDict
) -> Tuple[torch.Tensor, torch.Tensor]:
    X = X_test["X"]
    A = X_test["A"]
    Y = X_test["Y"]
    pi_X = torch.sigmoid(X.mv(theta["propensity_weights"]))
    mu_X = (
        X.mv(theta["outcome_weights"])
        + A * theta["treatment_weight"]
        + theta["intercept"]
    )
    analytic_eif_at_test_pts = (A / pi_X - (1 - A) / (1 - pi_X)) * (Y - mu_X)
    analytic_correction = analytic_eif_at_test_pts.mean()
    return analytic_correction, analytic_eif_at_test_pts
