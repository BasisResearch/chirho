from typing import Callable, Optional, Tuple

import functools
import torch
import math
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import pyro
import pyro.distributions as dist
from pyro.infer import Predictive

from chirho.counterfactual.handlers import MultiWorldCounterfactual
from chirho.indexed.ops import IndexSet, gather
from chirho.interventional.handlers import do
from chirho.robust.internals.utils import ParamDict
from chirho.robust.handlers.estimators import one_step_correction
from chirho.robust.handlers.predictive import PredictiveModel

pyro.settings.set(module_local_params=True)

sns.set_style("white")

pyro.set_rng_seed(321)  # for reproducibility


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
        X: torch.Tensor,
        A: torch.Tensor,
        Y: torch.Tensor,
        link_fn: Callable[..., dist.Distribution] = lambda mu: dist.Normal(mu, 1.0),
        prior_scale: Optional[float] = None,
    ):
        p = X.shape[1]
        super().__init__(p, link_fn, prior_scale)
        self.X = X
        self.A = A
        self.Y = Y

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


class GroundTruthModel(CausalGLM):
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
        self.beta = beta  # sparisty of outcome weights
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


class ATEFunctional(torch.nn.Module):
    def __init__(self, model: Callable, *, num_monte_carlo: int = 100):
        super().__init__()
        self.model = model
        self.num_monte_carlo = num_monte_carlo

    def forward(self, *args, **kwargs):
        with MultiWorldCounterfactual():
            with pyro.plate(
                "monte_carlo_functional", size=self.num_monte_carlo, dim=-2
            ):
                with do(actions=dict(A=(torch.tensor(0.0), torch.tensor(1.0)))):
                    Ys = self.model(*args, **kwargs)
                Y0 = gather(Ys, IndexSet(A={1}), event_dim=0)
                Y1 = gather(Ys, IndexSet(A={2}), event_dim=0)
        ate = (Y1 - Y0).mean(dim=-2, keepdim=True).mean(dim=-1, keepdim=True).squeeze()
        return pyro.deterministic("ATE", ate)


# Closed form expression
def closed_form_doubly_robust_ate_correction(
    X_test, theta
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


# Helper class to create a trivial guide that returns the maximum likelihood estimate
class MLEGuide(torch.nn.Module):
    def __init__(self, mle_est: ParamDict):
        super().__init__()
        self.names = list(mle_est.keys())
        for name, value in mle_est.items():
            setattr(self, name + "_param", torch.nn.Parameter(value))

    def forward(self, *args, **kwargs):
        for name in self.names:
            value = getattr(self, name + "_param")
            pyro.sample(
                name, pyro.distributions.Delta(value, event_dim=len(value.shape))
            )


def linear_kernel(X, Z=None):
    if Z == None:
        Z = X
    if len(X.shape) == 1:
        X = X[None, ...]
    if len(Z.shape) == 1:
        Z = Z[None, ...]
    return torch.einsum("...np,...mp->...nm", X, Z)[..., None, :, :]


class CausalGP(torch.nn.Module):
    def __init__(
        self,
        X_train: torch.Tensor,
        A_train: torch.Tensor,
        Y_train: torch.Tensor,
        kernel,
        noise: float = 1.0,
        prior_scale: Optional[float] = None,
    ):
        super().__init__()
        self.XA_train = torch.concatenate([X_train, A_train.unsqueeze(-1)], dim=-1)
        self.p = X_train.shape[1]
        self.kernel = kernel
        self.noise = noise
        alpha, K_inv = self._gp_mean_inverse(self.XA_train, Y_train)
        self.alpha = torch.nn.Parameter(alpha)
        self.K_inv = K_inv
        if prior_scale is None:
            self.prior_scale = 1 / math.sqrt(self.p)
        else:
            self.prior_scale = prior_scale

    def _gp_mean_inverse(self, X: torch.Tensor, Y: torch.Tensor):
        N = X.size(0)
        Kff = self.kernel(X)
        Kff.view(-1)[:: N + 1] += self.noise  # add noise to diagonal
        # Lff = torch.linalg.cholesky(Kff)
        # alpha = torch.cholesky_solve(Y.unsqueeze(-1), Lff).squeeze()
        K_inv = torch.cholesky_inverse(Kff)
        alpha = torch.einsum("...mn,...n->...m", K_inv, Y).squeeze()
        return alpha, K_inv

    def sample_covariate_loc_scale(self):
        return torch.zeros(self.p), torch.ones(self.p)

    def sample_propensity_weights(self):
        return pyro.sample(
            "propensity_weights",
            dist.Normal(0.0, self.prior_scale).expand((self.p,)).to_event(1),
        )

    def forward(self):
        x_loc, x_scale = self.sample_covariate_loc_scale()
        X = pyro.sample("X", dist.Normal(x_loc, x_scale).to_event(1))
        propensity_weights = self.sample_propensity_weights()
        A = pyro.sample(
            "A",
            dist.Bernoulli(
                logits=torch.einsum("...i,...i->...", X, propensity_weights)
            ),
        )
        # Sample Y from GP
        X = X.expand(A.shape + X.shape[-1:])
        XA = torch.concatenate([X, A.unsqueeze(-1)], dim=-1)
        # XA = XA.unsqueeze(0)
        f_loc = torch.einsum(
            "...mn,...n->...m", self.kernel(XA, self.XA_train), self.alpha
        )
        # cov_train_x = torch.cholesky_solve(self.kernel(self.XA_train, XA), self.Lff)
        cov_train_x = torch.einsum(
            "...tn,...nk->...tk", self.K_inv, self.kernel(self.XA_train, XA)
        )
        f_cov = self.kernel(XA) - torch.einsum(
            "...mn,...nk->...mk", self.kernel(XA, self.XA_train), cov_train_x
        )
        # return pyro.sample("Y", dist.MultivariateNormal(f_loc, f_cov))[..., 0, :]p
        return pyro.sample(
            "Y",
            dist.Normal(
                f_loc[..., 0, :], torch.diagonal(f_cov, dim1=-1, dim2=-2)[..., 0, :]
            ),
        )


N_datasets = 1
simulated_datasets = []

# Data configuration
p = 200
alpha = 50
beta = 50
N_train = 500
N_test = 500

true_model = GroundTruthModel(p, alpha, beta)

for _ in range(N_datasets):
    # Generate data
    D_train = Predictive(
        true_model, num_samples=N_train, return_sites=["X", "A", "Y"]
    )()
    D_test = Predictive(true_model, num_samples=N_test, return_sites=["X", "A", "Y"])()
    simulated_datasets.append((D_train, D_test))


fitted_params = []
for i in range(N_datasets):
    # Generate data
    D_train = simulated_datasets[i][0]

    # Fit model using maximum likelihood
    conditioned_model = ConditionedCausalGLM(
        X=D_train["X"], A=D_train["A"], Y=D_train["Y"]
    )

    guide_train = pyro.infer.autoguide.AutoDelta(conditioned_model)
    elbo = pyro.infer.Trace_ELBO()(conditioned_model, guide_train)

    # initialize parameters
    elbo()
    adam = torch.optim.Adam(elbo.parameters(), lr=0.03)

    # Do gradient steps
    for _ in range(2000):
        adam.zero_grad()
        loss = elbo()
        loss.backward()
        adam.step()

    theta_hat = {
        k: v.clone().detach().requires_grad_(True) for k, v in guide_train().items()
    }
    fitted_params.append(theta_hat)

# Compute doubly robust ATE estimates using both the automated and closed form expressions
plug_in_ates = []
analytic_corrections = []
automated_monte_carlo_corrections = []
for i in range(N_datasets):
    theta_hat = fitted_params[i]
    D_train = simulated_datasets[i][0]
    D_test = simulated_datasets[i][1]
    functional = functools.partial(ATEFunctional, num_monte_carlo=500)
    # gp_kernel = gp.kernels.RBF(input_dim=D_train['X'].shape[1] + 1)
    gp_kernel = linear_kernel
    fitted_gp_model = CausalGP(
        X_train=D_train["X"],
        A_train=D_train["A"],
        Y_train=D_train["Y"],
        kernel=gp_kernel,
    )
    ate_plug_in = functional(fitted_gp_model)()
    (
        analytic_correction,
        analytic_eif_at_test_pts,
    ) = closed_form_doubly_robust_ate_correction(D_test, theta_hat)
    print("here")
    automated_monte_carlo_correction = one_step_correction(
        fitted_gp_model,
        functional,
        num_samples_outer=max(10000, 100 * p),
        num_samples_inner=1,
    )(D_test)

    plug_in_ates.append(ate_plug_in.detach().item())
    analytic_corrections.append(analytic_correction.detach().item())
    automated_monte_carlo_corrections.append(
        automated_monte_carlo_correction.detach().item()
    )

plug_in_ates = np.array(plug_in_ates)
analytic_corrections = np.array(analytic_corrections)
automated_monte_carlo_corrections = np.array(automated_monte_carlo_corrections)
