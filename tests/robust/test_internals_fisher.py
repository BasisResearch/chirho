import math
from typing import TypeVar

import pyro
import pyro.distributions as dist
import pytest
import torch
from pyro.infer.predictive import Predictive

from chirho.robust.internals.linearize import make_empirical_fisher_vp
from chirho.robust.internals.utils import ParamDict
from chirho.robust.ops import Point

pyro.settings.set(module_local_params=True)


T = TypeVar("T")


class GaussianModel(pyro.nn.PyroModule):
    def __init__(self, cov_mat: torch.Tensor):
        super().__init__()
        self.register_buffer("cov_mat", cov_mat)

    def forward(self, loc):
        pyro.sample(
            "x", dist.MultivariateNormal(loc=loc, covariance_matrix=self.cov_mat)
        )


# Note: this is separate from the GaussianModel above because of upstream obstacles in the interaction between
# `pyro.nn.PyroModule` and `torch.func`. See https://github.com/BasisResearch/chirho/issues/393
def gaussian_log_prob(params: ParamDict, data_point: Point[T], cov_mat) -> T:
    with pyro.validation_enabled(False):
        return dist.MultivariateNormal(
            loc=params["loc"], covariance_matrix=cov_mat
        ).log_prob(data_point["x"])


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
def test_empirical_fisher_vp(loc: torch.Tensor, cov_mat: torch.Tensor, v: torch.Tensor):
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
