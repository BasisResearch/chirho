import time
from functools import partial

import pytest
import torch
from pyro.infer.predictive import Predictive

from chirho.robust.internals.linearize import make_empirical_fisher_vp

from .robust_fixtures import (
    GaussianModel,
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
