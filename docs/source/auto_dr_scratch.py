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
        self, N: int, p: int, link_fn: Callable[..., dist.Distribution] = gaussian_link
    ):
        super().__init__()
        self.N = N
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

    def sample_covariate_loc_scale(self):
        loc = pyro.sample(
            "covariate_loc", dist.Normal(0.0, 1.0).expand((self.p,)).to_event(1)
        )
        scale = pyro.sample(
            "covariate_scale", dist.LogNormal(0, 1).expand((self.p,)).to_event(1)
        )
        return loc, scale

    def forward(self):
        outcome_weights = self.sample_outcome_weights()
        propensity_weights = self.sample_propensity_weights()
        tau = self.sample_treatment_weight()
        x_loc, x_scale = self.sample_covariate_loc_scale()
        with pyro.plate("obs", self.N, dim=-1):
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
                    torch.einsum("...np,...p->...n", X, outcome_weights) + A * tau
                ),
            )


class KnownCovariateDistModel(HighDimLinearModel):
    def sample_covariate_loc_scale(self):
        return torch.zeros(self.p), torch.ones(self.p)


class BenchmarkLinearModel(HighDimLinearModel):
    def __init__(self, N: int, p: int, link_fn: Callable, alpha: int, beta: int):
        super().__init__(N, p, link_fn)
        self.alpha = alpha  # sparsity of propensity weights
        self.beta = beta  # sparisty of outcome weights

    def sample_outcome_weights(self):
        outcome_weights = 1 / math.sqrt(self.beta) * torch.ones(self.p)
        outcome_weights[self.beta :] = 0.0
        return outcome_weights

    def sample_propensity_weights(self):
        propensity_weights = 1 / math.sqrt(4 * self.alpha) * torch.ones(self.p)
        propensity_weights[self.alpha :] = 0.0
        return propensity_weights

    def sample_treatment_weight(self):
        return torch.tensor(0.0)

    def sample_covariate_loc_scale(self):
        return torch.zeros(self.p), torch.ones(self.p)


# Genrate data
p = 1
alpha = 1
beta = 1
N_train = 5
N_test = N_train  # 500  # TODO refactor model and ATE to not require N_test == N_train
benchmark_model_train = BenchmarkLinearModel(N_train, p, gaussian_link, alpha, beta)
benchmark_model_test = BenchmarkLinearModel(N_test, p, gaussian_link, alpha, beta)

with pyro.poutine.trace() as train_tr:
    benchmark_model_train()

with pyro.poutine.trace() as test_tr:
    benchmark_model_test()

D_train = {k: train_tr.trace.nodes[k]["value"] for k in ["X", "A", "Y"]}
D_test = {k: test_tr.trace.nodes[k]["value"] for k in ["X", "A", "Y"]}


# Fit MLE
class ConditionModelTrain(KnownCovariateDistModel):
    def forward(self):
        with condition(data=D_train):
            return super().forward()


model_train = ConditionModelTrain(N_train, p, gaussian_link)
guide_train = pyro.infer.autoguide.AutoDelta(model_train)
elbo = pyro.infer.Trace_ELBO()(model_train, guide_train)

# initialize parameters
elbo()

adam = torch.optim.Adam(elbo.parameters(), lr=0.03)

# Do gradient steps
for step in range(2000):
    adam.zero_grad()
    loss = elbo()
    loss.backward()
    adam.step()
    if step % 250 == 0:
        print("[iteration %04d] loss: %.4f" % (step, loss))


theta_hat = {
    k: v.clone().detach().requires_grad_(True) for k, v in guide_train().items()
}

# Double check gradient of log likelihood at theta_hat is close to zero
# with pyro.poutine.trace() as tr:
#     # with condition(data=theta_hat):
#     ConditionModelTrain(N_train, p, gaussian_link)()

# Compute EIF from Kennedy
X_ex = torch.tensor([[1.12345]])
A_ex = torch.tensor([1.0])
Y_ex = torch.tensor([2.213])

pi_X = torch.sigmoid(X_ex * theta_hat["propensity_weights"])
mu_X = (
    X_ex.squeeze() * theta_hat["outcome_weights"] + A_ex * theta_hat["treatment_weight"]
)

kennedy_correction = A_ex / pi_X * (Y_ex - mu_X)

print(kennedy_correction.mean())

# Compute EIF via fisher info formula
N_monte_carlo = 5e4
model = KnownCovariateDistModel(N_monte_carlo, p, gaussian_link)
theta_hat = collections.OrderedDict((k, theta_hat[k]) for k in sorted(theta_hat.keys()))
model_theta_hat = condition(data=theta_hat)(model)

D_dummy = {"X": X_ex, "A": A_ex, "Y": Y_ex}
model_theta_hat_test = condition(data=D_dummy)(
    condition(data=theta_hat)(KnownCovariateDistModel(1, p, gaussian_link))
)
log_likelihood_test = pyro.poutine.trace(model_theta_hat_test).get_trace()

log_likelihood_test.log_prob_sum()
log_likelihood_test = (
    log_likelihood_test.nodes["X"]["log_prob_sum"]
    + log_likelihood_test.nodes["A"]["log_prob_sum"]
    + log_likelihood_test.nodes["Y"]["log_prob_sum"]
) / D_dummy[next(iter(D_dummy))].shape[0]


scores = collections.OrderedDict(
    zip(theta_hat.keys(), torch.autograd.grad(log_likelihood_test, theta_hat.values()))
)


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


# Simulate a bunch of data from the model to compute fisher info
with pyro.poutine.trace() as fisher_tr:
    model_theta_hat()

D_model = {k: fisher_tr.trace.nodes[k]["value"] for k in ["X", "A", "Y"]}


def _f_hess(flat_theta: torch.tensor) -> torch.tensor:
    theta = unflatten_dict(flat_theta, theta_hat)
    model_theta_hat_fisher = condition(data=D_model)(
        condition(data=theta)(KnownCovariateDistModel(N_monte_carlo, p, gaussian_link))
    )
    log_likelihood_fisher = (
        pyro.poutine.trace(model_theta_hat_fisher).get_trace().log_prob_sum()
        / D_model[next(iter(D_model))].shape[0]
    )
    return log_likelihood_fisher


def fisher_info_mat(flat_theta: torch.tensor) -> torch.tensor:
    fisher_mat = torch.zeros((len(flat_theta), len(flat_theta)))
    for n in range(int(N_monte_carlo)):
        theta = unflatten_dict(flat_theta, theta_hat)
        # Get nth datapoint
        D_datapoint = {
            "X": D_model["X"][[n]],
            "A": D_model["A"][[n]],
            "Y": D_model["Y"][[n]],
        }
        model_theta_hat_fisher = condition(data=D_datapoint)(
            condition(data=theta)(KnownCovariateDistModel(1, p, gaussian_link))
        )

        log_like_trace = pyro.poutine.trace(model_theta_hat_fisher).get_trace()
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


# fisher_info_approx = torch.autograd.functional.hessian(_f_hess, flatten_dict(theta_hat))

fisher_info_approx = fisher_info_mat(flatten_dict(theta_hat))

inverse_fisher_info = torch.inverse(fisher_info_approx)

# This should match the kennedy formula
print(
    torch.einsum(
        "i,ij,j->",
        torch.tensor([0.0, 0.0, 1.0]),
        inverse_fisher_info,
        flatten_dict(scores),
    )
)


# plug_in = ATE_4(model_theta_hat) + (
#     0 * sum(theta_hat[k].sum() for k in theta_hat.keys())
# )
# plug_in_grads = collections.OrderedDict(
#     zip(theta_hat.keys(), torch.autograd.grad(plug_in, theta_hat.values()))
# )
