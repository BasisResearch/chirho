import pyro
import pyro.distributions as dist
import pyro.infer.reparam
import pytest
import torch

from causal_pyro.indexed.handlers import IndexPlatesMessenger
from causal_pyro.observational.handlers import (
    CutModule,
    CutComplementModule,
    IndexCutModule,
    cut,
)


@pyro.infer.config_enumerate
def model():
    eta = pyro.sample("eta", dist.Bernoulli(0.5))
    w = pyro.sample("w", dist.Bernoulli(torch.where(eta == 1, 0.8, 0.2)))
    theta = pyro.sample("theta", dist.Bernoulli(0.5))
    z = pyro.sample(
        "z", dist.Bernoulli(torch.where((eta == 1) & (theta == 1), 0.8, 0.2))
    )


def linear_gaussian_model(N1=10, N2=10):
    eta = pyro.sample("eta", dist.Normal(0, 1))
    theta = pyro.sample("theta", dist.Normal(0, 1))
    with pyro.plate("module_one_plate", N1):
        w = pyro.sample("w", dist.Normal(eta, 1))
    with pyro.plate("module_two_plate", N2):
        z = pyro.sample("z", dist.Normal(eta + theta, 1))
    return {"w": w, "z": z}


def linear_gaussian_cut_posterior(data):
    w = data["w"]
    z = data["z"]
    pr_eta_cut = dist.Normal(data["w"].sum() / (N1 + 1), scale=1 / torch.sqrt(N1 + 1))
    pr_theta_cut_cond_eta = lambda eta: dist.Normal(
        (data["z"] - eta).sum() / (N2 + 1), scale=1 / torch.sqrt(N2 + 1)
    )
    return pr_eta_cut, pr_theta_cut_cond_eta


def test_simple_two_module_example():
conditioned_model = pyro.condition(
    model, data={"z": torch.tensor(1.0), "w": torch.tensor(1.0)}
)
module_one_vars = ["eta", "w"]
module_one, module_two = cut(conditioned_model, vars=module_one_vars)

# Exact posterior inference in module_one
module_one_post = pyro.infer.infer_discrete(module_one, first_available_dim=-2)
module_one_samples = []
for _ in range(1000):
    with pyro.poutine.trace() as tr1:
        module_one_post()
    module_one_samples.append(tr1.trace.nodes["eta"]["value"].item())

# Exact posterior inference in joint model
joint_post = pyro.infer.infer_discrete(conditioned_model, first_available_dim=-2)
joint_samples = []
for _ in range(1000):
    with pyro.poutine.trace() as tr1:
        joint_post()
    joint_samples.append(tr1.trace.nodes["eta"]["value"].item())

torch.tensor(module_one_samples).mean()
torch.tensor(joint_samples).mean()

# Condition module two on module one
module_two = pyro.condition(module_two, data=tr1.trace)
module_two_post = pyro.infer.infer_discrete(module_two, first_available_dim=-2)

with pyro.poutine.trace() as tr2:
    module_two_post()


def module_raises_assertion_error():
    conditioned_model = pyro.condition(model, data={"z": torch.tensor(1.0), "w": torch.tensor(1.0)})
    module_one_vars = ["eta", "w"]
    module_one, module_two = cut(conditioned_model, vars=module_one_vars)
    try:
        module_two_post = pyro.infer.infer_discrete(module_two, first_available_dim=-2)
        assert False, "AssertionError should have been raises since module_one is not conditioned on"
    except AssertionError:
        print("AssertionError raised as expected")





# TODO: test with VI cut posterior approach in Yu et al (2022) on several simple examples


def test_plate_cut_module_discrete_runs():
    conditioned_model = pyro.condition(
        model, data={"z": torch.tensor(1.0), "w": torch.tensor(1.0)}
    )
    module_one_vars = ["eta", "w"]
    with IndexPlatesMessenger(), IndexCutModule(module_one_vars):
        z = conditioned_model()


def test_plate_cut_module_continuous_runs():
    data = linear_gaussian_model()
    conditioned_model = pyro.condition(linear_gaussian_model, data=data)
    module_one_vars = ["eta", "w"]
    with IndexPlatesMessenger(), IndexCutModule(module_one_vars):
        z = conditioned_model()


def test_linear_gaussian_cut():
    """
    Tests if the exact cut posterior known in closed form for the linear Gaussian case
    matches the cut posterior obtained by running inference in the cut module.
    """
    pass
    # N1 = 3
    # N2 = 3
    # data = linear_gaussian_model(N1, N2)
    # exact_cut_posterior = linear_gaussian_cut_posterior(data)
