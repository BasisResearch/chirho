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


@pytest.mark.parametrize(
    "data_config",
    [
        (torch.zeros(1, requires_grad=True), torch.eye(1)),
        (torch.ones(2, requires_grad=True), torch.eye(2)),
    ],
)
def test_fisher_vmap_smoke(data_config):
    loc, cov_mat = data_config
    func_log_prob = gaussian_log_prob
    log_prob_params = {"loc": loc}
    N_monte_carlo = 10000
    data = Predictive(GaussianModel(cov_mat), num_samples=N_monte_carlo)(loc)
    empirical_fisher_vp_func = make_empirical_fisher_vp(
        func_log_prob, log_prob_params, data, cov_mat=cov_mat
    )
    v_single = torch.ones(cov_mat.shape[1])
    v_batch = torch.stack([v_single, v_single], axis=0)
    empirical_fisher_vp_func_batched = torch.func.vmap(empirical_fisher_vp_func)

    # Check if fisher vector product works on a single vector and a batch of vectors
    empirical_fisher_vp_func({"loc": v_single})
    empirical_fisher_vp_func_batched({"loc": v_batch})
    try:
        empirical_fisher_vp_func({"loc": v_batch})
        assert False, "Fisher vector product should not work on a batch of vectors"
    except RuntimeError:
        pass
    try:
        empirical_fisher_vp_func_batched({"loc": v_single})
        assert False, "Batched Fisher vector product should not work on a single vector"
    except RuntimeError:
        pass


@pytest.mark.parametrize(
    "data_config",
    [
        (torch.zeros(1, requires_grad=True), torch.eye(1)),
        (torch.ones(2, requires_grad=True), torch.eye(2)),
    ],
)
def test_fisher_grad_smoke(data_config):
    loc, cov_mat = data_config
    func_log_prob = gaussian_log_prob
    log_prob_params = {"loc": loc}
    N_monte_carlo = 10000
    data = Predictive(GaussianModel(cov_mat), num_samples=N_monte_carlo)(loc)
    empirical_fisher_vp_func = make_empirical_fisher_vp(
        func_log_prob, log_prob_params, data, cov_mat=cov_mat
    )
    v = 0.5 * torch.ones(cov_mat.shape[1], requires_grad=True)
    f = lambda x: empirical_fisher_vp_func({"loc": x**2})["loc"].sum()

    # Check using `torch.func.grad`
    assert (
        torch.func.grad(f)(v).sum() != 0
    ), "Zero gradients but expected non-zero gradients"

    # Check using autograd
    assert torch.autograd.gradcheck(
        f, v, atol=0.2
    ), f"Finite difference gradients do not match autograd gradients"
