from typing import ParamSpec, TypeVar

import pyro
import pyro.distributions as dist
import pytest
import torch

from chirho.robust.ops import influence_fn

pyro.settings.set(module_local_params=True)

P = ParamSpec("P")
Q = ParamSpec("Q")
S = TypeVar("S")
T = TypeVar("T")


class SimpleModel(pyro.nn.PyroModule):
    def forward(self):
        a = pyro.sample("a", dist.Normal(0, 1))
        with pyro.plate("data", 3, dim=-1):
            b = pyro.sample("b", dist.Normal(a, 1))
            return pyro.sample("y", dist.Normal(a + b, 1))


class SimpleGuide(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.loc_a = torch.nn.Parameter(torch.rand(()))
        self.loc_b = torch.nn.Parameter(torch.rand((3,)))

    def forward(self):
        a = pyro.sample("a", dist.Normal(self.loc_a, 1.0))
        with pyro.plate("data", 3, dim=-1):
            b = pyro.sample("b", dist.Normal(self.loc_b, 1.0))
            return {"a": a, "b": b}


@pytest.mark.parametrize("model,guide", [(SimpleModel(), SimpleGuide())])
def test_nmc_influence_smoke(model, guide):
    num_samples_outer = 100
    param_eif = influence_fn(
        model,
        guide,
        max_plate_nesting=1,
        num_samples_outer=num_samples_outer,
    )

    with torch.no_grad():
        test_datum = {
            k: v[0]
            for k, v in pyro.infer.Predictive(
                model, num_samples=2, return_sites=["y"], parallel=True
            )().items()
        }

    print(test_datum, param_eif(test_datum))


@pytest.mark.parametrize("model,guide", [(SimpleModel(), SimpleGuide())])
def test_nmc_influence_vmap_smoke(model, guide):
    num_samples_outer = 100
    param_eif = influence_fn(
        model,
        guide,
        max_plate_nesting=1,
        num_samples_outer=num_samples_outer,
    )

    with torch.no_grad():
        test_data = pyro.infer.Predictive(
            model, num_samples=4, return_sites=["y"], parallel=True
        )()

    batch_param_eif = torch.vmap(param_eif, randomness="different")
    print(test_data, batch_param_eif(test_data))
