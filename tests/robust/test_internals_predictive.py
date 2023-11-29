import collections
from typing import Any, Callable, List, Set, Tuple, TypeVar

import pyro
import pyro.distributions as dist
import pytest
import torch
from typing_extensions import ParamSpec

from chirho.robust.internals.predictive import NMCLogPredictiveLikelihood

pyro.settings.set(module_local_params=True)

P = ParamSpec("P")
Q = ParamSpec("Q")
S = TypeVar("S")
T = TypeVar("T")


class SimpleModel(pyro.nn.PyroModule):
    def __init__(self):
        super().__init__()
        self.loc_a = torch.nn.Parameter(torch.rand(()))
        self.loc_b = torch.nn.Parameter(torch.rand((3,)))

    def forward(self, use_rsample: bool):
        Normal = (
            dist.Normal if use_rsample else dist.testing.fakes.NonreparameterizedNormal
        )
        a = pyro.sample("a", Normal(self.loc_a, 1))
        with pyro.plate("data", 3, dim=-1):
            b = pyro.sample("b", Normal(self.loc_b, 1))
            return pyro.sample("y", Normal(a + b, 1))


ModelTestCase = Tuple[
    Callable[[], Callable[[bool], Any]],
    Callable[[Callable[[bool], Any]], Callable[[bool], Any]],
    Set[str],
]

MODEL_TEST_CASES: List[ModelTestCase] = [
    (SimpleModel, lambda _: lambda *args: None, {"y"}),
]


@pytest.mark.parametrize("model,guide,obs_names", MODEL_TEST_CASES)
def test_grad_nmc_log_prob(model, guide, obs_names):
    num_samples = 10000

    model = model()
    guide = guide(pyro.poutine.block(hide=obs_names)(model))
    model(True), guide(True)  # initialize

    log_prob = NMCLogPredictiveLikelihood(model, guide, num_samples=num_samples)
    params = collections.OrderedDict(log_prob.named_parameters())

    with torch.no_grad():
        test_datum = {
            k: v[0]
            for k, v in pyro.infer.Predictive(
                model, num_samples=2, return_sites=obs_names, parallel=True
            )(True).items()
        }

    test_datum_log_prob_reparam = log_prob(test_datum, True)
    grad_test_datum_log_prob_reparam = dict(
        zip(
            params.keys(),
            torch.autograd.grad(test_datum_log_prob_reparam, params.values()),
        )
    )

    test_datum_log_prob_score = log_prob(test_datum, False)
    grad_test_datum_log_prob_score = dict(
        zip(
            params.keys(),
            torch.autograd.grad(test_datum_log_prob_score, params.values()),
        )
    )

    for k in params.keys():
        expected_grad = grad_test_datum_log_prob_reparam[k]
        actual_grad = grad_test_datum_log_prob_score[k]
        assert torch.allclose(actual_grad, expected_grad)
