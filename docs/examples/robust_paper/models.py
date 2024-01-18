from typing import Callable, Optional, Dict
import torch
import math
import pyro
import pyro.distributions as dist

pyro.settings.set(module_local_params=True)


class CausalGLM(pyro.nn.PyroModule):
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

        self.observed_sites = ["X", "A", "Y"]

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
        return torch.zeros(self.p), torch.ones(self.p)

    def forward(self):
        intercept = self.sample_intercept()
        outcome_weights = self.sample_outcome_weights()
        propensity_weights = self.sample_propensity_weights()
        tau = self.sample_treatment_weight()
        x_loc, x_scale = self.sample_covariate_loc_scale()
        X = pyro.sample("X", dist.Normal(x_loc, x_scale).to_event(1))
        A = pyro.sample(
            "A",
            dist.Bernoulli(
                logits=torch.einsum("...i,...i->...", X, propensity_weights)
            ),
        )

        return pyro.sample(
            "Y",
            self.link_fn(
                torch.einsum("...i,...i->...", X, outcome_weights) + A * tau + intercept
            ),
        )


class ConditionedCausalGLM(CausalGLM):
    def __init__(
        self,
        data: Dict[str, torch.Tensor],
        *,
        link_fn: Callable[..., dist.Distribution] = lambda mu: dist.Normal(mu, 1.0),
        prior_scale: Optional[float] = None,
    ):
        p = X.shape[1]
        super().__init__(p, link_fn, prior_scale)
        self.X = data["X"]
        self.A = data["A"]
        self.Y = data["Y"]

    def forward(self):
        intercept = self.sample_intercept()
        outcome_weights = self.sample_outcome_weights()
        propensity_weights = self.sample_propensity_weights()
        tau = self.sample_treatment_weight()
        x_loc, x_scale = self.sample_covariate_loc_scale()
        with pyro.plate("__train__", size=self.X.shape[0], dim=-1):
            X = pyro.sample("X", dist.Normal(x_loc, x_scale).to_event(1), obs=self.X)
            A = pyro.sample(
                "A",
                dist.Bernoulli(
                    logits=torch.einsum("ni,i->n", self.X, propensity_weights)
                ),
                obs=self.A,
            )
            pyro.sample(
                "Y",
                self.link_fn(
                    torch.einsum("ni,i->n", X, outcome_weights) + A * tau + intercept
                ),
                obs=self.Y,
            )


class DataGeneratorCausalGLM(CausalGLM):
    def __init__(
        self,
        p: int,
        alpha: int,
        beta: int,
        link_fn: Callable[..., dist.Distribution] = lambda mu: dist.Normal(mu, 1.0),
        treatment_weight: float = 0.0,
    ):
        super().__init__(p, link_fn)
        self.alpha = alpha  # sparsity of propensity weights
        self.beta = beta  # sparsity of outcome weights
        self.treatment_weight = treatment_weight

    def sample_outcome_weights(self):
        outcome_weights = 1 / math.sqrt(self.beta) * torch.ones(self.p)
        outcome_weights[self.beta :] = 0.0
        return outcome_weights

    def sample_propensity_weights(self):
        propensity_weights = 1 / math.sqrt(self.alpha) * torch.ones(self.p)
        propensity_weights[self.alpha :] = 0.0
        return propensity_weights

    def sample_treatment_weight(self):
        return torch.tensor(self.treatment_weight)

    def sample_intercept(self):
        return torch.tensor(0.0)


class MultivariateNormalModel(pyro.nn.PyroModule):
    def __init__(self, p: int):
        super().__init__()
        self.p = p
        self.observed_sites = ["x"]

    def sample_mean(self):
        return pyro.sample("mu", dist.Normal(0.0, 1.0).expand((self.p,)).to_event(1))

    def sample_scale_tril(self):
        if self.p > 1:
            return pyro.sample("scale_tril", dist.LKJCholesky(self.p))
        else:
            return pyro.sample(
                "scale_tril", dist.HalfNormal(1.0).expand((self.p, self.p)).to_event(1)
            )

    def forward(self) -> torch.Tensor:
        mu = self.sample_mean()
        scale_tril = self.sample_scale_tril()
        return pyro.sample("x", dist.MultivariateNormal(loc=mu, scale_tril=scale_tril))


class ConditionedMultivariateNormalModel(MultivariateNormalModel):
    def __init__(self, data: Dict[str, torch.Tensor], *, p: int):
        super().__init__(p)
        self.x = data["x"]

    def forward(self):
        mu = self.sample_mean()
        scale_tril = self.sample_scale_tril()
        with pyro.plate("__train__", size=self.x.shape[0], dim=-1):
            pyro.sample(
                "x",
                dist.MultivariateNormal(loc=mu, scale_tril=scale_tril),
                obs=self.x,
            )


class DataGeneratorMultivariateNormalModel(MultivariateNormalModel):
    def sample_mean(self):
        return torch.zeros(self.p)

    def sample_scale_tril(self):
        return torch.eye(self.p)


class kernel_ridge:
    pass


class neural_network:
    pass
