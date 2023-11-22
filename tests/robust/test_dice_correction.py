import collections
import math
from functools import partial
from typing import Callable, Dict, List, Optional

import pyro
import pyro.distributions as dist
import torch

from chirho.robust import one_step_correction
from chirho.robust.functionals import average_treatment_effect, dice_correction
from chirho.robust.internals.utils import _flatten_dict, _unflatten_dict


class HighDimLinearModel(pyro.nn.PyroModule):
    def __init__(
        self,
        p: int,
        link_fn: Callable[..., dist.Distribution] = lambda mu: dist.Normal(mu, 1.0),
        prior_scale: float = None,
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

    def forward(self, N: int):
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
                ).mask(False),
            )
            return pyro.sample(
                "Y",
                self.link_fn(
                    torch.einsum("...np,...p->...n", X, outcome_weights)
                    + A * tau
                    + intercept
                ),
            )


# Internal structure of the model (that's given)
# and then outer monte carlo samples


class KnownCovariateDistModel(HighDimLinearModel):
    def sample_covariate_loc_scale(self):
        return torch.zeros(self.p), torch.ones(self.p)


class FakeNormal(dist.Normal):
    has_rsample = False


def test_bernoulli_model():
    p = 1
    n_monte_carlo_outer = 1000
    avg_plug_in_grads = torch.zeros(4)
    for _ in range(n_monte_carlo_outer):
        n_monte_carlo_inner = 100
        target_functional = partial(
            dice_correction(average_treatment_effect), n_monte_carlo=n_monte_carlo_inner
        )
        # bernoulli_link = lambda mu: dist.Bernoulli(logits=mu)
        link = lambda mu: FakeNormal(mu, 1.0)
        # link = lambda mu: dist.Normal(mu, 1.0)
        model = KnownCovariateDistModel(p, link)
        theta_hat = {
            "intercept": torch.tensor(0.0).requires_grad_(True),
            "outcome_weights": torch.tensor([1.0]).requires_grad_(True),
            "propensity_weights": torch.tensor([1.0]).requires_grad_(True),
            "treatment_weight": torch.tensor(1.0).requires_grad_(True),
        }

        # Canonical ordering of parameters when flattening and unflattening
        theta_hat = collections.OrderedDict(
            (k, theta_hat[k]) for k in sorted(theta_hat.keys())
        )
        flat_theta = _flatten_dict(theta_hat)

        # Compute gradient of plug-in functional
        plug_in = target_functional(model, theta_hat)
        plug_in += (
            0 * flat_theta.sum()
        )  # hack for full gradient (maintain flattened shape)

        avg_plug_in_grads += (
            _flatten_dict(
                collections.OrderedDict(
                    zip(
                        theta_hat.keys(),
                        torch.autograd.grad(plug_in, theta_hat.values()),
                    )
                )
            )
            / n_monte_carlo_outer
        )

    correct_grad = torch.tensor([0, 0, 0, 1.0])
    # assert (avg_plug_in_grads - correct_grad).abs().sum() < 1 / torch.sqrt(
    #     torch.tensor(n_monte_carlo_outer)
    # )
