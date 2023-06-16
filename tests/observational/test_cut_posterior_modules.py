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


def test_simple_two_module_example():
    conditioned_model = pyro.condition(
        model, data={"z": torch.tensor(1.0), "w": torch.tensor(1.0)}
    )
    module_one_vars = ["eta", "w"]
    module_one, module_two = cut(conditioned_model, vars=module_one_vars)

    # Exact posterior inference in module_one
    module_one_post = pyro.infer.infer_discrete(module_one, first_available_dim=-2)
    with pyro.poutine.trace() as tr1:
        module_one_post()

    # Condition module two on module one
    module_two = pyro.condition(module_two, data=tr1.trace)
    module_two_post = pyro.infer.infer_discrete(module_two, first_available_dim=-2)

    with pyro.poutine.trace() as tr2:
        module_two_post()


# TODO: test with VI cut posterior approach in Yu et al (2022) on several simple examples


def test_plate_cut_module():
    conditioned_model = pyro.condition(
        model, data={"z": torch.tensor(1.0), "w": torch.tensor(1.0)}
    )
    module_one_vars = ["eta", "w"]
    with IndexPlatesMessenger(), IndexCutModule(module_one_vars):
        conditioned_model()
