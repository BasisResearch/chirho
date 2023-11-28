import pyro
import pytest
import torch
from pyro.infer.predictive import Predictive
import pyro.distributions as dist
from .robust_fixtures import (
    GaussianModel,
    gaussian_log_prob,
)

from chirho.robust.internals.linearize import make_empirical_fisher_vp, linearize

pyro.settings.set(module_local_params=True)


@pytest.mark.parametrize(
    "loc", [torch.zeros(2, requires_grad=True), torch.ones(2, requires_grad=True)]
)
@pytest.mark.parametrize(
    "cov_mat",
    [
        torch.eye(2, requires_grad=False),
        torch.tensor(torch.ones(2, 2) + torch.eye(2), requires_grad=False),
    ],
)
@pytest.mark.parametrize(
    "v",
    [
        torch.tensor([1.0, 0.0], requires_grad=False),
        torch.tensor([0.0, 1.0], requires_grad=False),
        torch.tensor([1.0, 1.0], requires_grad=False),
        torch.tensor([0.0, 0.0], requires_grad=False),
    ],
)
def test_empirical_fisher_vp_against_analytical(
    loc: torch.Tensor, cov_mat: torch.Tensor, v: torch.Tensor
):
    func_log_prob = gaussian_log_prob
    log_prob_params = {"loc": loc}
    N_monte_carlo = 10000
    data = Predictive(GaussianModel(cov_mat), num_samples=N_monte_carlo)(loc)
    empirical_fisher_vp_func = make_empirical_fisher_vp(
        func_log_prob, log_prob_params, data, cov_mat=cov_mat
    )

    empirical_fisher_vp = empirical_fisher_vp_func({"loc": v})["loc"]

    prec_matrix = torch.linalg.inv(cov_mat)
    true_vp = prec_matrix.mv(v)

    assert torch.all(torch.isclose(empirical_fisher_vp, true_vp, atol=0.1))
