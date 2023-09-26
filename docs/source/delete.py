import collections
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

# pyro.set_rng_seed(321) # for reproducibility

gaussian_link = lambda mu: dist.Normal(mu, 1.0)
bernoulli_link = lambda mu: dist.Bernoulli(logits=mu)


class HighDimLinearModel(pyro.nn.PyroModule):
    def __init__(
        self, p: int, link_fn: Callable[..., dist.Distribution] = gaussian_link
    ):
        super().__init__()
        self.p = p
        self.link_fn = link_fn

    def sample_outcome_weights(self):
        return pyro.sample(
            "outcome_weights",
            dist.Normal(0.0, 1.0 / math.sqrt(self.p)).expand((self.p,)).to_event(1),
        )

    def sample_propensity_weights(self):
        return pyro.sample(
            "propensity_weights",
            dist.Normal(0.0, 1.0 / math.sqrt(self.p)).expand((self.p,)).to_event(1),
        )

    def sample_treatment_weight(self):
        return pyro.sample("treatment_weight", dist.Normal(0.0, 1.0))

    def sample_treatment_null_weight(self):
        return pyro.sample("treatment_null_weight", dist.Normal(0.0, 1.0))

    def sample_covariate_loc_scale(self):
        loc = pyro.sample(
            "covariate_loc", dist.Normal(0.0, 1.0).expand((self.p,)).to_event(1)
        )
        scale = pyro.sample(
            "covariate_scale", dist.LogNormal(0, 1).expand((self.p,)).to_event(1)
        )
        return loc, scale

    def forward(self, N: int):
        outcome_weights = self.sample_outcome_weights()
        propensity_weights = self.sample_propensity_weights()
        tau = self.sample_treatment_weight()
        tau0 = self.sample_treatment_null_weight()
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
                    + (1 - A) * tau0
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

    def sample_propensity_weights(self):
        propensity_weights = 1 / math.sqrt(4 * self.alpha) * torch.ones(self.p)
        propensity_weights[self.alpha :] = 0.0
        return propensity_weights

    def sample_treatment_weight(self):
        return torch.tensor(self.treatment_weight)

    def sample_treatment_null_weight(self):
        return torch.tensor(0.0)

    def sample_covariate_loc_scale(self):
        return torch.zeros(self.p), torch.ones(self.p)


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


def fisher_info_mat_slow(conditioned_model, flat_theta, D_model) -> torch.tensor:
    N_monte_carlo = D_model[next(iter(D_model))].shape[0]
    fisher_mat = torch.zeros((len(flat_theta), len(flat_theta)))
    for n in range(int(N_monte_carlo)):
        theta = unflatten_dict(flat_theta, theta_hat)
        model_theta_hat_conditioned = condition(data=theta)(conditioned_model)
        # Get nth datapoint
        D_datapoint = {
            "X": D_model["X"][[n]],
            "A": D_model["A"][[n]],
            "Y": D_model["Y"][[n]],
        }

        log_like_trace = pyro.poutine.trace(model_theta_hat_conditioned).get_trace(
            D_datapoint
        )
        log_like_trace.log_prob_sum()
        log_likelihood_fisher = (
            log_like_trace.nodes["X"]["log_prob_sum"]
            + log_like_trace.nodes["A"]["log_prob_sum"]
            + log_like_trace.nodes["Y"]["log_prob_sum"]
        )

        scores_datapoint = collections.OrderedDict(
            zip(
                theta.keys(),
                torch.autograd.grad(log_likelihood_fisher, theta.values()),
            )
        )
        # print(pyro.poutine.trace(model_theta_hat_fisher).get_trace())
        # return pyro.poutine.trace(model_theta_hat_fisher).get_trace()
        scores_datapoint = flatten_dict(scores_datapoint)
        fisher_mat += 1 / N_monte_carlo * scores_datapoint.outer(scores_datapoint)
    return fisher_mat


def monte_carlo_fisher_info_of_model(
    unconditioned_model: Callable[[], torch.tensor],  # simulates data
    conditioned_model: Callable[[], torch.tensor],  # computes log likelihood
    theta_hat: Dict[str, torch.tensor],
    obs_names: List[str],
    N_monte_carlo: int = None,
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
    # import pdb

    # pdb.set_trace()
    # Generate N_monte_carlo samples from the model
    with pyro.poutine.trace() as model_tr:
        model_theta_hat_unconditioned(N=N_monte_carlo)
    D_model = {k: model_tr.trace.nodes[k]["value"] for k in obs_names}

    # Compute fisher information matrix from these samples
    def _log_prob_at_datapoints(flat_theta: torch.tensor):
        # Need to duplicate conditioning on theta for pytorch to register gradients (TODO: any fix?)
        theta = unflatten_dict(flat_theta, theta_hat)
        model_theta_hat_conditioned = condition(data=theta)(conditioned_model)
        log_like_trace = pyro.poutine.trace(model_theta_hat_conditioned).get_trace(
            D_model
        )
        log_like_trace.compute_log_prob()
        log_prob_at_datapoints = torch.zeros(N_monte_carlo)
        for name in obs_names:
            log_prob_at_datapoints += log_like_trace.nodes[name]["log_prob"]

        return log_prob_at_datapoints

    log_prob_grads = torch.autograd.functional.jacobian(
        _log_prob_at_datapoints, flat_theta
    )

    assert log_prob_grads.shape[0] == N_monte_carlo
    assert log_prob_grads.shape[1] == theta_dim
    return 1 / N_monte_carlo * log_prob_grads.T.mm(log_prob_grads)


def one_step_correction(
    target_functional: Callable[[Callable], torch.tensor],
    unconditioned_model: Callable[[], torch.tensor],  # simulates data
    conditioned_model: Callable[[], torch.tensor],  # computes log likelihood
    obs_names: List[str],
    theta_hat: Dict[str, torch.tensor],
    X_test: Dict[str, torch.tensor],
    *,
    eps_fisher: float = 1e-8,
    N_monte_carlo: int = None
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
    )  # hack to make sure we get full gradient vector

    plug_in_grads = flatten_dict(
        collections.OrderedDict(
            zip(
                theta_hat.keys(),
                torch.autograd.grad(plug_in, theta_hat.values()),
            )
        )
    )

    # compute the score function for the new data
    log_likelihood_test = pyro.poutine.trace(model_theta_hat_conditioned).get_trace(
        X_test
    )
    log_likelihood_test.log_prob_sum()
    log_likelihood_test = (
        log_likelihood_test.nodes["X"]["log_prob_sum"]
        + log_likelihood_test.nodes["A"]["log_prob_sum"]
        + log_likelihood_test.nodes["Y"]["log_prob_sum"]
    ) / X_test[next(iter(X_test))].shape[0]

    scores = flatten_dict(
        collections.OrderedDict(
            zip(
                theta_hat.keys(),
                torch.autograd.grad(log_likelihood_test, theta_hat.values()),
            )
        )
    )

    # TODO: remove this
    ####################
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

    log_prob_grads_at_test = torch.autograd.functional.jacobian(
        _log_prob_at_datapoints, flat_theta
    )

    with pyro.poutine.trace() as model_tr:
        model_theta_hat_unconditioned(N=N_monte_carlo)
    D_model = {k: model_tr.trace.nodes[k]["value"] for k in obs_names}
    fisher_slow = fisher_info_mat_slow(conditioned_model, flat_theta, D_model)
    inverse_slow = torch.inverse(fisher_slow)
    ###################

    # compute inverse fisher information matrix
    fisher_info_approx = monte_carlo_fisher_info_of_model(
        unconditioned_model, conditioned_model, theta_hat, obs_names, N_monte_carlo
    )
    inverse_fisher_info = torch.inverse(
        fisher_info_approx + eps_fisher * torch.eye(fisher_info_approx.shape[0])
    )

    # compute the correction
    import pdb

    correction_scores = torch.einsum(
        "i,ij,jk->k", plug_in_grads, inverse_fisher_info, log_prob_grads_at_test.T
    )
    (torch.abs(correction_scores - kennedy_correction_vec)).max()

    correction_scores_slow = torch.einsum(
        "i,ij,jk->k", plug_in_grads, inverse_slow, log_prob_grads_at_test.T
    )

    pdb.set_trace()

    return torch.einsum("i,ij,j->", plug_in_grads, inverse_fisher_info, scores)


p = 1
alpha = 1
beta = 1
N_train = 250
N_test = 250
benchmark_model = BenchmarkLinearModel(p, gaussian_link, alpha, beta)

with pyro.poutine.trace() as train_tr:
    benchmark_model(N=N_train)

with pyro.poutine.trace() as test_tr:
    benchmark_model(N=N_test)

D_train = {k: train_tr.trace.nodes[k]["value"] for k in ["X", "A", "Y"]}
D_test = {k: test_tr.trace.nodes[k]["value"] for k in ["X", "A", "Y"]}

# X_ex = torch.tensor([[1.12345]])
# A_ex = torch.tensor([1.0])
# Y_ex = torch.tensor([2.213])
# N_test = 1
# D_test = {"X": X_ex, "A": A_ex, "Y": Y_ex}


# Fit model to training data (uncorrected)
class ConditionedModel(KnownCovariateDistModel):
    def forward(self, D):
        with condition(data=D):
            # Assume first dimension corresponds to # of datapoints
            N = D[next(iter(D))].shape[0]
            return super().forward(N=N)


conditioned_model = ConditionedModel(p, gaussian_link)
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
    if step % 250 == 0:
        print("[iteration %04d] loss: %.4f" % (step, loss))

theta_hat = {
    k: v.clone().detach().requires_grad_(True) for k, v in guide_train().items()
}
print(theta_hat.keys(), theta_hat["treatment_weight"])


def ATE(model: Callable[[], torch.Tensor], num_samples: int = 100) -> torch.Tensor:
    """Compute the average treatment effect of a model."""

    @pyro.plate("num_samples", num_samples, dim=-2)
    def _ate_model():
        with MultiWorldCounterfactual():
            with do(actions=dict(A=(torch.tensor(0.0), torch.tensor(1.0)))):
                Ys = model()
            Y0 = gather(Ys, IndexSet(A={1}), event_dim=0)
            Y1 = gather(Ys, IndexSet(A={2}), event_dim=0)
            return pyro.deterministic("ATE", (Y1 - Y0).mean(dim=-1, keepdim=True))

    return _ate_model().mean(dim=-2, keepdim=True).squeeze()


# TODO: this will incur monte-carlo error
def ATE_2(model: Callable[[], torch.Tensor], num_samples: int = 100) -> torch.Tensor:
    """Compute the average treatment effect of a model."""
    with do(actions=dict(A=torch.tensor(0.0))):
        Y0 = model(N=num_samples)
    with do(actions=dict(A=torch.tensor(1.0))):
        Y1 = model(N=num_samples)

    return pyro.deterministic("ATE", (Y1 - Y0).mean(dim=-1, keepdim=True))


# TODO: this will not incur much monte-carlo error since X is fixed across worlds
def ATE_3(model: Callable[[], torch.Tensor], num_samples: int = 100) -> torch.Tensor:
    """Compute the average treatment effect of a model."""
    with MultiWorldCounterfactual():
        with do(actions=dict(A=(torch.tensor(0.0), torch.tensor(1.0)))):
            Ys = model(N=num_samples)
        Y0 = gather(Ys, IndexSet(A={1}), event_dim=0)
        Y1 = gather(Ys, IndexSet(A={2}), event_dim=0)
    return pyro.deterministic("ATE", (Y1 - Y0).mean(dim=-1, keepdim=True).squeeze())


# # You can't modify joint; all queries need to be some function of the joint without
# # modifying it??
# def ATE_4(model: Callable[[], torch.Tensor], num_samples: int = 100) -> torch.Tensor:
#     """Compute the average treatment effect of a model."""
#     with pyro.poutine.trace() as data_tr:
#         model(N=num_samples)

#     Y = data_tr.trace.nodes["Y"]["value"]
#     A = data_tr.trace.nodes["A"]["value"]
#     Y1_expectation = Y[A == 1].mean(dim=-1, keepdim=True).squeeze()
#     Y0_expectation = Y[A == 0].mean(dim=-1, keepdim=True).squeeze()
#     return pyro.deterministic("ATE", Y1_expectation - Y0_expectation)


def closed_form_correction(X_test, theta):
    X = X_test["X"]
    A = X_test["A"]
    Y = X_test["Y"]
    # Compute EIF from Kennedy
    pi_X = torch.sigmoid(X.mv(theta["propensity_weights"]))
    mu_X = X.mv(theta["outcome_weights"]) + A * theta["treatment_weight"]

    kennedy_correction = (A / pi_X - (1 - A) / (1 - pi_X)) * (Y - mu_X)
    return kennedy_correction.mean(), kennedy_correction


unconditioned_model = KnownCovariateDistModel(p, gaussian_link)
model_cond_theta = condition(data=theta_hat)(unconditioned_model)

ATE_plugin = ATE_3(model_cond_theta, num_samples=10000)

print("ATE plugin", ATE_plugin)
kennedy_correction, kennedy_correction_vec = closed_form_correction(D_test, theta_hat)

print("Kennedy closed-form correction", kennedy_correction)

ATE_correction = one_step_correction(
    lambda m: ATE_3(m, num_samples=1000),
    unconditioned_model,
    conditioned_model,
    ["X", "A", "Y"],
    theta_hat,
    D_test,
    eps_fisher=0.0,
    N_monte_carlo=int(1e5),
)
ATE_onestep = ATE_plugin + ATE_correction
print(ATE_plugin, ATE_correction, ATE_onestep)
