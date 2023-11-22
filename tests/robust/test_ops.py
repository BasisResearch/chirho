import functools
from typing import Callable, List, Mapping, Optional, ParamSpec, Set, Tuple, TypeVar

import pyro
import pyro.distributions as dist
import pytest
import torch

from chirho.robust.internals.predictive import PredictiveFunctional
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
        a = pyro.sample("a", dist.Normal(self.loc_a, 1))
        with pyro.plate("data", 3, dim=-1):
            b = pyro.sample("b", dist.Normal(self.loc_b, 1))
            return {"a": a, "b": b}


MODEL_TEST_CASES: List[Tuple[Callable, Callable, Set[str], Optional[int]]] = [
    (SimpleModel(), SimpleGuide(), {"y"}, 1),
    (SimpleModel(), SimpleGuide(), {"y"}, None),
    pytest.param(
        (m := SimpleModel()),
        pyro.infer.autoguide.AutoNormal(m),
        {"y"},
        1,
        marks=pytest.mark.xfail(
            reason="torch.func autograd doesnt work with PyroParam"
        ),
    ),
    pytest.param(
        (m := SimpleModel()),
        pyro.infer.autoguide.AutoDelta(m),
        {"y"},
        1,
        marks=pytest.mark.xfail(
            reason="torch.func autograd doesnt work with PyroParam"
        ),
    ),
]


@pytest.mark.parametrize("model,guide,obs_names,max_plate_nesting", MODEL_TEST_CASES)
@pytest.mark.parametrize(
    "num_samples_outer,num_samples_inner", [(100, None), (10, 100)]
)
@pytest.mark.parametrize("cg_iters", [None, 1, 10])
@pytest.mark.parametrize("num_predictive_samples", [1, 5])
def test_nmc_predictive_influence_smoke(
    model,
    guide,
    obs_names,
    max_plate_nesting,
    num_samples_outer,
    num_samples_inner,
    cg_iters,
    num_predictive_samples,
):
    model(), guide()  # initialize

    predictive_eif = influence_fn(
        model,
        guide,
        functional=functools.partial(
            PredictiveFunctional, num_samples=num_predictive_samples
        ),
        max_plate_nesting=max_plate_nesting,
        num_samples_outer=num_samples_outer,
        num_samples_inner=num_samples_inner,
        cg_iters=cg_iters,
    )

    with torch.no_grad():
        test_datum = {
            k: v[0]
            for k, v in pyro.infer.Predictive(
                model, num_samples=2, return_sites=obs_names, parallel=True
            )().items()
        }

    test_datum_eif: Mapping[str, torch.Tensor] = predictive_eif(test_datum)
    assert len(test_datum_eif) > 0
    for k, v in test_datum_eif.items():
        assert not torch.isnan(v).any(), f"eif for {k} had nans"
        assert not torch.isinf(v).any(), f"eif for {k} had infs"
        assert not torch.isclose(v, torch.zeros_like(v)).all(), f"eif for {k} was zero"


@pytest.mark.parametrize("model,guide,obs_names,max_plate_nesting", MODEL_TEST_CASES)
@pytest.mark.parametrize(
    "num_samples_outer,num_samples_inner", [(100, None), (10, 100)]
)
@pytest.mark.parametrize("cg_iters", [None, 1, 10])
@pytest.mark.parametrize("num_predictive_samples", [1, 5])
def test_nmc_predictive_influence_vmap_smoke(
    model,
    guide,
    obs_names,
    max_plate_nesting,
    num_samples_outer,
    num_samples_inner,
    cg_iters,
    num_predictive_samples,
):
    model(), guide()  # initialize

    predictive_eif = influence_fn(
        model,
        guide,
        functional=functools.partial(
            PredictiveFunctional, num_samples=num_predictive_samples
        ),
        max_plate_nesting=max_plate_nesting,
        num_samples_outer=num_samples_outer,
        num_samples_inner=num_samples_inner,
        cg_iters=cg_iters,
    )

    with torch.no_grad():
        test_data = pyro.infer.Predictive(
            model, num_samples=4, return_sites=obs_names, parallel=True
        )()

    batch_predictive_eif = torch.vmap(predictive_eif, randomness="different")
    test_data_eif: Mapping[str, torch.Tensor] = batch_predictive_eif(test_data)
    assert len(test_data_eif) > 0
    for k, v in test_data_eif.items():
        assert not torch.isnan(v).any(), f"eif for {k} had nans"
        assert not torch.isinf(v).any(), f"eif for {k} had infs"
        assert not torch.isclose(v, torch.zeros_like(v)).all(), f"eif for {k} was zero"
