import math
import time
import warnings
from functools import partial
from typing import Any, Callable, Container, Generic, Optional, TypeVar

import pyro
import pytest
import torch
from typing_extensions import ParamSpec

from chirho.indexed.handlers import DependentMaskMessenger
from chirho.observational.handlers import condition
from chirho.robust.handlers.predictive import PredictiveModel
from chirho.robust.internals.linearize import make_empirical_fisher_vp
from chirho.robust.internals.nmc import BatchedNMCLogMarginalLikelihood
from chirho.robust.internals.utils import guess_max_plate_nesting, make_functional_call
from chirho.robust.ops import Point

from .robust_fixtures import SimpleGuide, SimpleModel

pyro.settings.set(module_local_params=True)

P = ParamSpec("P")
Q = ParamSpec("Q")
S = TypeVar("S")
T = TypeVar("T")


class _UnmaskNamedSites(DependentMaskMessenger):
    names: Container[str]

    def __init__(self, names: Container[str]):
        self.names = names

    def get_mask(
        self,
        dist: pyro.distributions.Distribution,
        value: Optional[torch.Tensor],
        device: torch.device = torch.device("cpu"),
        name: Optional[str] = None,
    ) -> torch.Tensor:
        return torch.tensor(name is None or name in self.names, device=device)


class OldNMCLogPredictiveLikelihood(Generic[P, T], torch.nn.Module):
    model: Callable[P, Any]
    guide: Callable[P, Any]
    num_samples: int
    max_plate_nesting: Optional[int]

    def __init__(
        self,
        model: torch.nn.Module,
        guide: torch.nn.Module,
        *,
        num_samples: int = 1,
        max_plate_nesting: Optional[int] = None,
    ):
        super().__init__()
        self.model = model
        self.guide = guide
        self.num_samples = num_samples
        self.max_plate_nesting = max_plate_nesting

    def forward(
        self, data: Point[T], *args: P.args, **kwargs: P.kwargs
    ) -> torch.Tensor:
        if self.max_plate_nesting is None:
            self.max_plate_nesting = guess_max_plate_nesting(
                self.model, self.guide, *args, **kwargs
            )
            warnings.warn(
                "Since max_plate_nesting is not specified, \
                the first call to NMCLogPredictiveLikelihood will not be seeded properly. \
                See https://github.com/BasisResearch/chirho/pull/408"
            )

        masked_guide = pyro.poutine.mask(mask=False)(self.guide)
        masked_model = _UnmaskNamedSites(names=set(data.keys()))(
            condition(data=data)(self.model)
        )
        log_weights = pyro.infer.importance.vectorized_importance_weights(
            masked_model,
            masked_guide,
            *args,
            num_samples=self.num_samples,
            max_plate_nesting=self.max_plate_nesting,
            **kwargs,
        )[0]
        return torch.logsumexp(log_weights, dim=0) - math.log(self.num_samples)


class SimpleMultivariateGaussianModel(pyro.nn.PyroModule):
    def __init__(self, p):
        super().__init__()
        self.p = p

    def forward(self):
        loc = pyro.sample(
            "loc", pyro.distributions.Normal(torch.zeros(self.p), 1.0).to_event(1)
        )
        cov_mat = torch.eye(self.p)
        return pyro.sample("y", pyro.distributions.MultivariateNormal(loc, cov_mat))


class SimpleMultivariateGuide(torch.nn.Module):
    def __init__(self, p):
        super().__init__()
        self.loc_ = torch.nn.Parameter(torch.rand((p,)))
        self.p = p

    def forward(self):
        return pyro.sample("loc", pyro.distributions.Normal(self.loc_, 1).to_event(1))


model_guide_types = [
    (
        partial(SimpleMultivariateGaussianModel, p=500),
        partial(SimpleMultivariateGuide, p=500),
    ),
    (SimpleModel, SimpleGuide),
]


@pytest.mark.skip(reason="This test is too slow to run on CI")
@pytest.mark.parametrize("model_guide", model_guide_types)
def test_empirical_fisher_vp_performance_with_likelihood(model_guide):
    num_monte_carlo = 10000
    model_family, guide_family = model_guide

    model = model_family()
    guide = guide_family()

    model()
    guide()

    start_time = time.time()
    data = pyro.infer.Predictive(
        model, guide=guide, num_samples=num_monte_carlo, return_sites=["y"]
    )()
    end_time = time.time()
    print("Data generation time (s): ", end_time - start_time)

    log1_prob_params, func1_log_prob = make_functional_call(
        OldNMCLogPredictiveLikelihood(model, guide, max_plate_nesting=1)
    )
    batched_func1_log_prob = torch.func.vmap(
        func1_log_prob, in_dims=(None, 0), randomness="different"
    )

    log2_prob_params, func2_log_prob = make_functional_call(
        BatchedNMCLogMarginalLikelihood(PredictiveModel(model, guide))
    )

    fisher_hessian_vmapped = make_empirical_fisher_vp(
        batched_func1_log_prob, log1_prob_params, data
    )

    fisher_hessian_batched = make_empirical_fisher_vp(
        func2_log_prob, log2_prob_params, data
    )

    v1 = {
        k: torch.ones_like(v) if k != "guide.loc_a" else torch.zeros_like(v)
        for k, v in log1_prob_params.items()
    }
    v2 = {f"model.{k}": v for k, v in v1.items()}

    func2_log_prob(log2_prob_params, data)

    start_time = time.time()
    fisher_hessian_vmapped(v1)
    end_time = time.time()
    print("Hessian vmapped time (s): ", end_time - start_time)

    start_time = time.time()
    fisher_hessian_batched(v2)
    end_time = time.time()
    print("Hessian manual batched time (s): ", end_time - start_time)
