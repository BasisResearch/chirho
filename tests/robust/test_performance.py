import time
from functools import partial

import pyro
import pytest
import torch
from pyro.infer.predictive import Predictive

from chirho.robust.internals.linearize import make_empirical_fisher_vp
from chirho.robust.internals.predictive import (
    NMCLogPredictiveLikelihood,
    PointLogPredictiveLikelihood,
)
from chirho.robust.internals.utils import make_functional_call

from .robust_fixtures import (
    GaussianModel,
    SimpleGuide,
    SimpleModel,
    _make_fisher_jvp_score_formulation,
    gaussian_log_prob,
    gaussian_log_prob_flattened,
)


@pytest.mark.skip(reason="This test is too slow to run on CI")
def test_empirical_fisher_vp_performance():
    p = 500
    loc = torch.tensor(1.0 * torch.ones(p), requires_grad=True)
    cov_mat = torch.eye(p)
    v = torch.ones(p)
    func_log_prob = gaussian_log_prob
    log_prob_params = {"loc": loc}
    num_monte_carlo = 10000
    start_time = time.time()
    data = Predictive(GaussianModel(cov_mat), num_samples=num_monte_carlo)(loc)
    end_time = time.time()
    print("Data generation time (s): ", end_time - start_time)

    fisher_hessian_vmapped = make_empirical_fisher_vp(
        func_log_prob, log_prob_params, data, is_batched=False, cov_mat=cov_mat
    )

    fisher_hessian_batched = make_empirical_fisher_vp(
        func_log_prob, log_prob_params, data, is_batched=True, cov_mat=cov_mat
    )

    f = partial(gaussian_log_prob_flattened, data_point=data, cov_mat=cov_mat)
    fisher_score_batched = _make_fisher_jvp_score_formulation(f, loc, num_monte_carlo)

    start_time = time.time()
    fisher_hessian_vmapped({"loc": v})
    end_time = time.time()
    print("Hessian vmapped time (s): ", end_time - start_time)

    start_time = time.time()
    fisher_hessian_batched({"loc": v})
    end_time = time.time()
    print("Hessian manual batched time (s): ", end_time - start_time)

    start_time = time.time()
    fisher_score_batched(v)
    end_time = time.time()
    print("Score manual batched time (s): ", end_time - start_time)


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
    (SimpleMultivariateGaussianModel, SimpleMultivariateGuide),
    (SimpleModel, SimpleGuide),
]


@pytest.mark.skip(reason="This test is too slow to run on CI")
@pytest.mark.parametrize("model_guide", model_guide_types)
def test_empirical_fisher_vp_performance_with_likelihood(model_guide):
    p = 500
    num_monte_carlo = 10000
    model_family, guide_family = model_guide

    model = model_family(p=p)
    guide = guide_family(p=p)

    model()
    guide()

    start_time = time.time()
    data = Predictive(
        model, guide=guide, num_samples=num_monte_carlo, return_sites=["y"]
    )()
    end_time = time.time()
    print("Data generation time (s): ", end_time - start_time)

    log1_prob_params, func1_log_prob = make_functional_call(
        NMCLogPredictiveLikelihood(model, guide)
    )

    log2_prob_params, func2_log_prob = make_functional_call(
        PointLogPredictiveLikelihood(model, guide)
    )

    fisher_hessian_vmapped = make_empirical_fisher_vp(
        func1_log_prob, log1_prob_params, data, is_batched=False
    )

    fisher_hessian_batched = make_empirical_fisher_vp(
        func2_log_prob, log2_prob_params, data, is_batched=True
    )

    v = {
        k: torch.ones_like(v) if k != "guide.loc_a" else torch.zeros_like(v)
        for k, v in log1_prob_params.items()
    }

    func2_log_prob(log2_prob_params, data)

    start_time = time.time()
    fisher_hessian_vmapped(v)
    end_time = time.time()
    print("Hessian vmapped time (s): ", end_time - start_time)

    start_time = time.time()
    fisher_hessian_batched(v)
    end_time = time.time()
    print("Hessian manual batched time (s): ", end_time - start_time)
