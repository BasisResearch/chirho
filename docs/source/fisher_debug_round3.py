import collections
from functools import partial
import math
import seaborn as sns
import matplotlib.pyplot as plt
from typing import Callable, Dict

import torch
import pyro
import pyro.distributions as dist
from pyro.infer.autoguide import AutoNormal
from pyro.infer import Predictive
from typing import Callable, Dict, List, Optional, Tuple, Union

from chirho.observational.handlers import condition
from chirho.interventional.handlers import do
from chirho.counterfactual.handlers import MultiWorldCounterfactual
from chirho.indexed.ops import IndexSet, gather

pyro.settings.set(module_local_params=True)

sns.set_style("white")

pyro.set_rng_seed(321)  # for reproducibility

gaussian_link = lambda mu: dist.Normal(mu, 1.0)
bernoulli_link = lambda mu: dist.Bernoulli(logits=mu)


class HighDimLinearModel(pyro.nn.PyroModule):
    def __init__(
        self,
        p: int,
        link_fn: Callable[..., dist.Distribution] = gaussian_link,
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
        link_fn: Callable,
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

    def sample_treatment_null_weight(self):
        return torch.tensor(0.0)

    def sample_covariate_loc_scale(self):
        return torch.zeros(self.p), torch.ones(self.p)


class ConditionedModel(KnownCovariateDistModel):
    def forward(self, D):
        with condition(data=D):
            # Assume first dimension corresponds to # of datapoints
            N = D[next(iter(D))].shape[0]
            return super().forward(N=N)


def ATE(model: Callable[[], torch.Tensor], num_samples: int = 100) -> torch.Tensor:
    """Compute the average treatment effect of a model."""
    with MultiWorldCounterfactual():
        with do(actions=dict(A=(torch.tensor(0.0), torch.tensor(1.0)))):
            Ys = model(N=num_samples)
        Y0 = gather(Ys, IndexSet(A={1}), event_dim=0)
        Y1 = gather(Ys, IndexSet(A={2}), event_dim=0)
    return pyro.deterministic("ATE", (Y1 - Y0).mean(dim=-1, keepdim=True).squeeze())


def make_log_likelihood_fn(
    model: Callable[[], torch.tensor],
    theta_hat: Dict[str, torch.tensor],
    X: Dict[str, torch.Tensor],
    obs_names: List[str],
) -> torch.tensor:
    n_samps = X[next(iter(X))].shape[0]

    def log_likelihood_fn(flat_theta: torch.tensor):
        theta = unflatten_dict(flat_theta, theta_hat)
        model_theta_hat_conditioned = condition(data=theta)(model)
        log_like_trace = pyro.poutine.trace(model_theta_hat_conditioned).get_trace(X)
        log_like_trace.compute_log_prob()
        log_prob_at_datapoints = torch.zeros(n_samps)
        for name in obs_names:
            log_prob_at_datapoints += log_like_trace.nodes[name]["log_prob"]
        return log_prob_at_datapoints

    return log_likelihood_fn


def make_log_likelihood_fn_two(
    model: Callable[[], torch.tensor],
    theta_hat: Dict[str, torch.tensor],
    obs_names: List[str],
) -> torch.tensor:
    def log_likelihood_fn(flat_theta: torch.tensor, X: Dict[str, torch.Tensor]):
        n_samps = X[next(iter(X))].shape[0]
        theta = unflatten_dict(flat_theta, theta_hat)
        model_theta_hat_conditioned = condition(data=theta)(model)
        log_like_trace = pyro.poutine.trace(model_theta_hat_conditioned).get_trace(X)
        log_like_trace.compute_log_prob()
        log_prob_at_datapoints = torch.zeros(n_samps)
        for name in obs_names:
            log_prob_at_datapoints += log_like_trace.nodes[name]["log_prob"]
        return log_prob_at_datapoints

    return log_likelihood_fn


def empirical_ihvp(
    f: Callable,
    theta: torch.tensor,
    v: torch.tensor,
    X: Dict[str, torch.tensor],
    twice_diff: bool = True,
    hessian_rescale: Optional[Union[float, None]] = None,
    n_max_hessian_samples: int = 25,
) -> torch.tensor:
    n_monte_carlo = X[next(iter(X))].shape[0]
    assert n_monte_carlo > 0

    if hessian_rescale is None:
        assert n_max_hessian_samples <= n_monte_carlo
        dim_latents = theta.shape[0]
        hessian_rescale = torch.tensor(1.0)
        # Approximate based on randomized max trace which provides high probability upper bound on spectral norm
        # (1) Sample random element of diagonal of Hessian
        # (2) Multiply by number of latents
        # (3) take max over n_max_hessian_samples
        for _ in range(n_max_hessian_samples):
            # Randomly sample a datapoint
            j = torch.randint(0, n_monte_carlo, (1,)).item()
            f_j = partial(f, X={key: val[[j]] for key, val in X.items()})

            # Randomly sample an element from the diagonal of the Hessian
            h_ix = torch.randint(0, dim_latents, (1,)).item()
            e_rand = torch.zeros(dim_latents)
            e_rand[h_ix] = 1.0
            if twice_diff:
                stochastic_trace = (
                    dim_latents
                    * -1
                    * torch.autograd.functional.vhp(f_j, theta, v=e_rand)[1].t()[h_ix]
                )
            else:
                stochastic_trace = (
                    dim_latents
                    * -1
                    * torch.autograd.functional.hvp(f_j, theta, v=e_rand)[1][h_ix]
                )
            if stochastic_trace > hessian_rescale:
                hessian_rescale = stochastic_trace

    assert hessian_rescale >= 1

    # If hessian rescale is very large, raise a warning
    if hessian_rescale > 1e6:
        print(
            f"Warning: hessian_rescale is large ({hessian_rescale}) and might lead to numerical instability."
        )

    ihvp = v
    for j in range(n_monte_carlo):
        # If twice differentiable, use VHP
        # https://pytorch.org/docs/stable/generated/torch.autograd.functional.hvp.html
        f_j = partial(f, X={key: val[[j]] for key, val in X.items()})
        if twice_diff:
            update = (
                v
                - torch.autograd.functional.vhp(f_j, theta, v=ihvp)[1].t()
                / hessian_rescale
            )
        else:
            update = (
                v
                - torch.autograd.functional.hvp(f_j, theta, v=ihvp)[1] / hessian_rescale
            )
        ihvp = ihvp + update
    import pdb

    pdb.set_trace()
    return -1 * ihvp / hessian_rescale  # Fisher information = -1 * E[hessian]


# TODO: change so that only need model
def inv_fisher_info_of_model_issa(
    v: torch.tensor,
    unconditioned_model: Callable[[], torch.tensor],  # simulates data
    conditioned_model: Callable[[], torch.tensor],  # computes log likelihood
    theta_hat: Dict[str, torch.tensor],
    obs_names: List[str],
    N_monte_carlo: int = None,
    hessian_rescale: Optional[Union[float, None]] = None,
) -> torch.tensor:
    """
    Compute the monte carlo estimate of the fisher information matrix.
    """
    flat_theta = flatten_dict(theta_hat)
    theta_dim = flat_theta.shape[0]
    model_theta_hat_unconditioned = condition(data=theta_hat)(unconditioned_model)
    if N_monte_carlo is None:
        N_monte_carlo = 25 * theta_dim  # 25 samples per parameter
    else:
        assert (
            N_monte_carlo >= theta_dim
        ), "N_monte_carlo must be at least as large as the number of parameters"
        if N_monte_carlo < 25 * theta_dim:
            print(
                "Warning: N_monte_carlo is less than 25 times the number of parameters. This may lead to inaccurate estimates."
            )

    # Generate N_monte_carlo samples from the model
    with pyro.poutine.trace() as model_tr:
        model_theta_hat_unconditioned(N=N_monte_carlo)
    D_model = {k: model_tr.trace.nodes[k]["value"] for k in obs_names}
    log_likelihood_fn = make_log_likelihood_fn_two(
        conditioned_model, theta_hat, obs_names
    )
    return empirical_ihvp(log_likelihood_fn, flat_theta, v, D_model)


def flatten_dict(d: Dict[str, torch.tensor]) -> torch.tensor:
    """
    Flatten a dictionary of tensors into a single vector.
    """
    return torch.cat([v.flatten() for k, v in d.items()])


def unflatten_dict(
    x: torch.tensor, d: Dict[str, torch.tensor]
) -> Dict[str, torch.tensor]:
    """
    Unflatten a vector into a dictionary of tensors.
    """
    return collections.OrderedDict(
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


def one_step_correction(
    target_functional: Callable[[Callable], torch.tensor],
    unconditioned_model: Callable[[], torch.tensor],  # simulates data
    conditioned_model: Callable[[], torch.tensor],  # computes log likelihood
    obs_names: List[str],
    theta_hat: Dict[str, torch.tensor],
    X_test: Dict[str, torch.tensor],
    *,
    all_scores: bool = False,
    eps_fisher: float = 1e-8,
    N_monte_carlo: int = None,
) -> torch.tensor:
    """
    One step correction for a given target functional.
    """
    theta_hat = collections.OrderedDict(
        (k, theta_hat[k]) for k in sorted(theta_hat.keys())
    )
    flat_theta = flatten_dict(theta_hat)
    model_theta_hat_unconditioned = condition(data=theta_hat)(unconditioned_model)
    model_theta_hat_conditioned = condition(data=theta_hat)(conditioned_model)

    plug_in = target_functional(model_theta_hat_unconditioned) + (
        0 * flat_theta.sum()
    )  # hack to make sure we get full gradient vector from autograd to maintain flattened gradient shapes

    plug_in_grads = flatten_dict(
        collections.OrderedDict(
            zip(
                theta_hat.keys(),
                torch.autograd.grad(plug_in, theta_hat.values()),
            )
        )
    )

    if all_scores:

        def _log_prob_at_datapoints(flat_theta: torch.tensor):
            # Need to duplicate conditioning on theta for pytorch to register gradients (TODO: any fix?)
            theta = unflatten_dict(flat_theta, theta_hat)
            model_theta_hat_conditioned = condition(data=theta)(conditioned_model)
            log_like_trace = pyro.poutine.trace(model_theta_hat_conditioned).get_trace(
                X_test
            )
            log_like_trace.compute_log_prob()
            log_prob_at_datapoints = torch.zeros(X_test[next(iter(X_test))].shape[0])
            for name in obs_names:
                log_prob_at_datapoints += log_like_trace.nodes[name]["log_prob"]

            return log_prob_at_datapoints

        scores = torch.autograd.functional.jacobian(_log_prob_at_datapoints, flat_theta)

    else:
        # compute the score function for new data
        N_test = X_test[next(iter(X_test))].shape[0]
        log_likelihood_test = pyro.poutine.trace(model_theta_hat_conditioned).get_trace(
            X_test
        )
        log_likelihood_test.log_prob_sum()
        log_prob_sum_test = torch.tensor(0.0)
        for name in obs_names:
            log_prob_sum_test += (
                log_likelihood_test.nodes[name]["log_prob_sum"] / N_test
            )

        scores = flatten_dict(
            collections.OrderedDict(
                zip(
                    theta_hat.keys(),
                    torch.autograd.grad(log_prob_sum_test, theta_hat.values()),
                )
            )
        )

    # # compute inverse fisher information matrix
    # fisher_info_approx = monte_carlo_fisher_info_of_model(
    #     unconditioned_model, conditioned_model, theta_hat, obs_names, N_monte_carlo
    # )
    # inverse_fisher_info = torch.inverse(
    #     fisher_info_approx + eps_fisher * torch.eye(fisher_info_approx.shape[0])
    # )
    a = inv_fisher_info_of_model_issa(
        scores,
        unconditioned_model,
        conditioned_model,
        theta_hat,
        obs_names,
        N_monte_carlo,
    )

    return plug_in_grads.dot(a)

    # # compute the correction
    # if all_scores:
    #     return torch.einsum("i,ij,jk->k", plug_in_grads, inverse_fisher_info, scores.T)
    # return torch.einsum("i,ij,j->", plug_in_grads, inverse_fisher_info, scores)


# Traditional ATE correction based on analytic derivation (see, for example, Kennedy (2023))
def closed_form_ate_correction(X_test, theta):
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


p = 1
alpha = 1
beta = 1
N_train = 10
N_test = 500
link = gaussian_link

pyro.clear_param_store()

# Generate data
benchmark_model = BenchmarkLinearModel(p, link, alpha, beta)

with pyro.poutine.trace() as train_tr:
    benchmark_model(N=N_train)

with pyro.poutine.trace() as test_tr:
    benchmark_model(N=N_test)

D_train = {k: train_tr.trace.nodes[k]["value"] for k in ["X", "A", "Y"]}
D_test = {k: test_tr.trace.nodes[k]["value"] for k in ["X", "A", "Y"]}

conditioned_model = ConditionedModel(p, link)
guide_train = pyro.infer.autoguide.AutoDelta(conditioned_model)
elbo = pyro.infer.Trace_ELBO()(conditioned_model, guide_train)

# initialize parameters
elbo(D_train)

adam = torch.optim.Adam(elbo.parameters(), lr=0.03)

# Do gradient steps
for step in range(2000):
    adam.zero_grad()
    loss = elbo(D_train)
    loss.backward()
    adam.step()
    # if step % 250 == 0:
    #     print("[iteration %04d] loss: %.4f" % (step, loss))
theta_hat = {
    k: v.clone().detach().requires_grad_(True) for k, v in guide_train().items()
}
print(theta_hat.keys(), theta_hat["treatment_weight"])

unconditioned_model = KnownCovariateDistModel(p, link)
model_cond_theta = condition(data=theta_hat)(unconditioned_model)

analytic_correction, analytic_eif_at_test_pts = closed_form_ate_correction(
    D_test, theta_hat
)

ATE_plugin = ATE(model_cond_theta, num_samples=10000)
# print("ATE plugin", ATE_plugin)

dim_latents = flatten_dict(theta_hat).shape[0]
N_monte_carlo_grid = [
    10 * dim_latents,
    100 * dim_latents,
    1000 * dim_latents,
    10000 * dim_latents,
    25000 * dim_latents,
]
relative_errors = []
absolute_errors = []
monte_correction = []
actual_correction = []
signed_relative_errors = []

for N_monte_carlo in N_monte_carlo_grid:
    print(f"N_monte_carlo = {N_monte_carlo}")
    ATE_correction = one_step_correction(
        lambda m: ATE(m, num_samples=1000),
        unconditioned_model,
        conditioned_model,
        ["X", "A", "Y"],
        theta_hat,
        D_test,
        eps_fisher=0.0,
        N_monte_carlo=N_monte_carlo,
        all_scores=False,
    )
    relative_errors.append(
        ((ATE_correction - analytic_eif_at_test_pts) / analytic_eif_at_test_pts).abs()
    )
    signed_relative_errors.append(
        ((ATE_correction - analytic_eif_at_test_pts) / analytic_eif_at_test_pts)
    )
    absolute_errors.append((ATE_correction - analytic_eif_at_test_pts).abs())
    monte_correction.append(ATE_correction)
    actual_correction.append(analytic_eif_at_test_pts)

    print((analytic_correction - ATE_correction).abs())
    print((analytic_correction - ATE_correction).abs() / analytic_correction.abs())
