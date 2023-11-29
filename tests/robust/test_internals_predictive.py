import collections
from typing import Callable, List, Mapping, Optional, Set, Tuple, TypeVar

import pyro
import pyro.distributions as dist
import pytest
import torch
from typing_extensions import ParamSpec

from chirho.robust.internals.predictive import (
    NMCLogPredictiveLikelihood,
    PredictiveFunctional,
)
from chirho.robust.internals.utils import make_functional_call

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


ModelTestCase = Tuple[
    Callable[[], Callable], Callable[[Callable], Callable], Set[str], Optional[int]
]

MODEL_TEST_CASES: List[ModelTestCase] = [
    (SimpleModel, lambda _: SimpleGuide(), {"y"}, 1),
    (SimpleModel, lambda _: SimpleGuide(), {"y"}, None),
    pytest.param(
        SimpleModel,
        pyro.infer.autoguide.AutoNormal,
        {"y"},
        1,
        marks=pytest.mark.xfail(
            reason="torch.func autograd doesnt work with PyroParam"
        ),
    ),
]


@pytest.mark.parametrize("model,guide,obs_names,max_plate_nesting", MODEL_TEST_CASES)
@pytest.mark.parametrize("num_samples", [10, 100])
def test_grad_nmc_log_prob(
    model,
    guide,
    obs_names,
    max_plate_nesting,
    num_samples,
):
    model = model()
    guide = guide(pyro.poutine.block(hide=obs_names)(model))

    model(), guide()  # initialize

    log_prob = NMCLogPredictiveLikelihood(
        model,
        guide,
        num_samples=num_samples,
        max_plate_nesting=max_plate_nesting,
    )

    with torch.no_grad():
        test_datum = {
            k: v[0]
            for k, v in pyro.infer.Predictive(
                model, num_samples=2, return_sites=obs_names, parallel=True
            )().items()
        }

    test_datum_log_prob = log_prob(test_datum)
    assert not torch.isnan(test_datum_log_prob)
    assert not torch.isinf(test_datum_log_prob)
    assert not torch.isclose(
        test_datum_log_prob, torch.zeros_like(test_datum_log_prob)
    ).all()

    params = collections.OrderedDict(log_prob.named_parameters())
    grad_test_datum_log_prob = dict(
        zip(params.keys(), torch.autograd.grad(test_datum_log_prob, params.values()))
    )

    assert len(grad_test_datum_log_prob) > 0
    for k, v in grad_test_datum_log_prob.items():
        assert v is not None, f"grad for {k} was None"
        assert not torch.isnan(v).any(), f"grad for {k} had nans"
        assert not torch.isinf(v).any(), f"grad for {k} had infs"
        assert not torch.isclose(v, torch.zeros_like(v)).all(), f"grad for {k} was zero"
