import functools
from typing import Callable, List, Mapping, Optional, Set, Tuple, TypeVar

import pyro
import pyro.distributions as dist
import pytest
import torch
from pyro.infer.predictive import Predictive
from typing_extensions import ParamSpec

from chirho.robust.handlers.predictive import PredictiveModel
from chirho.robust.internals.linearize import (
    conjugate_gradient_solve,
    linearize,
    make_empirical_fisher_vp,
)

from .robust_fixtures import (
    BenchmarkLinearModel,
    DataConditionedModel,
    GaussianModel,
    KnownCovariateDistModel,
    MLEGuide,
    SimpleGuide,
    SimpleModel,
    closed_form_ate_correction,
    gaussian_log_prob,
)

pyro.settings.set(module_local_params=True)

P = ParamSpec("P")
Q = ParamSpec("Q")
S = TypeVar("S")
T = TypeVar("T")


@pytest.mark.parametrize("ndim", [1, 2, 3, 10])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("num_particles", [1, 4])
def test_batch_cg_solve(ndim: int, dtype: torch.dtype, num_particles: int):
    cg_iters = None
    residual_tol = 1e-10

    U = torch.rand(ndim, ndim, dtype=dtype)
    A = torch.eye(ndim, dtype=dtype) + 0.1 * U.mm(U.t())
    expected_x = torch.randn(num_particles, ndim, dtype=dtype)
    b = torch.einsum("ij,nj->ni", A, expected_x)
    assert b.shape == (num_particles, ndim)

    batch_solve = functools.partial(
        conjugate_gradient_solve,
        lambda v: torch.einsum("ij,nj->ni", A, v),
        cg_iters=cg_iters,
        residual_tol=residual_tol,
    )

    actual_x = batch_solve(b)

    assert torch.all(torch.sum((actual_x - expected_x) ** 2, dim=1) < 1e-4)


ModelTestCase = Tuple[
    Callable[[], Callable], Callable[[Callable], Callable], Set[str], Optional[int]
]

MODEL_TEST_CASES: List[ModelTestCase] = [
    (SimpleModel, lambda _: SimpleGuide(), {"y"}, 1),
    (SimpleModel, lambda _: SimpleGuide(), {"y"}, None),
    pytest.param(
        SimpleModel,
        lambda m: pyro.infer.autoguide.AutoNormal(
            pyro.poutine.block(
                hide=[
                    "y",
                ]
            )(m)
        ),
        {"y"},
        1,
        marks=(
            [pytest.mark.xfail(reason="torch.func autograd doesnt work with PyroParam")]
            if tuple(map(int, pyro.__version__.split("+")[0].split(".")[:3]))
            <= (1, 8, 6)
            else []
        ),
    ),
]


@pytest.mark.parametrize("model,guide,obs_names,max_plate_nesting", MODEL_TEST_CASES)
@pytest.mark.parametrize("num_samples_outer,num_samples_inner", [(10, None), (10, 100)])
@pytest.mark.parametrize("cg_iters", [None, 1, 10])
def test_nmc_param_influence_smoke(
    model,
    guide,
    obs_names,
    max_plate_nesting,
    num_samples_outer,
    num_samples_inner,
    cg_iters,
):
    model = model()
    guide = guide(model)

    model(), guide()  # initialize

    param_eif = linearize(
        PredictiveModel(model, guide),
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

    test_datum_eif: Mapping[str, torch.Tensor] = param_eif(test_datum)
    assert len(test_datum_eif) > 0
    for k, v in test_datum_eif.items():
        assert not torch.isnan(v).any(), f"eif for {k} had nans"
        assert not torch.isinf(v).any(), f"eif for {k} had infs"
        if not (k.endswith("guide.loc_a") or k.endswith("a_unconstrained")):
            assert not torch.isclose(
                v, torch.zeros_like(v)
            ).all(), f"eif for {k} was zero"
        else:
            assert torch.isclose(
                v, torch.zeros_like(v)
            ).all(), f"eif for {k} should be zero"


@pytest.mark.parametrize("model,guide,obs_names,max_plate_nesting", MODEL_TEST_CASES)
@pytest.mark.parametrize("num_samples_outer,num_samples_inner", [(10, None), (10, 100)])
@pytest.mark.parametrize("cg_iters", [None, 1, 10])
def test_nmc_param_influence_vmap_smoke(
    model,
    guide,
    obs_names,
    max_plate_nesting,
    num_samples_outer,
    num_samples_inner,
    cg_iters,
):
    model = model()
    guide = guide(model)

    model(), guide()  # initialize

    param_eif = linearize(
        PredictiveModel(model, guide),
        max_plate_nesting=max_plate_nesting,
        num_samples_outer=num_samples_outer,
        num_samples_inner=num_samples_inner,
        cg_iters=cg_iters,
    )

    with torch.no_grad():
        test_data = pyro.infer.Predictive(
            model, num_samples=4, return_sites=obs_names, parallel=True
        )()

    test_data_eif: Mapping[str, torch.Tensor] = param_eif(test_data)
    assert len(test_data_eif) > 0
    for k, v in test_data_eif.items():
        assert not torch.isnan(v).any(), f"eif for {k} had nans"
        assert not torch.isinf(v).any(), f"eif for {k} had infs"
        if not (k.endswith("guide.loc_a") or k.endswith("a_unconstrained")):
            assert not torch.isclose(
                v, torch.zeros_like(v)
            ).all(), f"eif for {k} was zero"
        else:
            assert torch.isclose(
                v, torch.zeros_like(v)
            ).all(), f"eif for {k} should be zero"


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
def test_fisher_vmap(data_config):
    loc, cov_mat = data_config
    func_log_prob = gaussian_log_prob
    log_prob_params = {"loc": loc}
    N_monte_carlo = 10000
    data = Predictive(GaussianModel(cov_mat), num_samples=N_monte_carlo)(loc)
    empirical_fisher_vp_func = make_empirical_fisher_vp(
        func_log_prob, log_prob_params, data, cov_mat=cov_mat
    )
    v_single_one = torch.ones(cov_mat.shape[1])
    v_single_two = 0.4 * torch.ones(cov_mat.shape[1])
    v_batch = torch.stack([v_single_one, v_single_two], axis=0)
    empirical_fisher_vp_func_batched = torch.func.vmap(empirical_fisher_vp_func)

    # Check if fisher vector product works on a single vector and a batch of vectors
    single_one_out = empirical_fisher_vp_func({"loc": v_single_one})
    single_two_out = empirical_fisher_vp_func({"loc": v_single_two})
    batch_out = empirical_fisher_vp_func_batched({"loc": v_batch})

    assert torch.allclose(batch_out["loc"][0], single_one_out["loc"])
    assert torch.allclose(batch_out["loc"][1], single_two_out["loc"])

    with pytest.raises(RuntimeError):
        # Fisher vector product should not work on a batch of vectors
        empirical_fisher_vp_func({"loc": v_batch})
    with pytest.raises(RuntimeError):
        # Batched Fisher vector product should not work on a single vector
        empirical_fisher_vp_func_batched({"loc": v_single_one})


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

    def f(x):
        return empirical_fisher_vp_func({"loc": x})["loc"].sum()

    # Check using `torch.func.grad`
    assert (
        torch.func.grad(f)(v).sum() != 0
    ), "Zero gradients but expected non-zero gradients"

    # Check using autograd
    assert torch.autograd.gradcheck(
        f, v, atol=0.2
    ), "Finite difference gradients do not match autograd gradients"


def test_linearize_against_analytic_ate():
    p = 1
    alpha = 1
    beta = 1
    N_train = 100
    N_test = 100

    def link(mu):
        return dist.Normal(mu, 1.0)

    # Generate data
    benchmark_model = BenchmarkLinearModel(p, link, alpha, beta)
    D_train = Predictive(
        benchmark_model, num_samples=N_train, return_sites=["X", "A", "Y"]
    )()
    D_train = {k: v.squeeze(-1) for k, v in D_train.items()}
    D_test = Predictive(
        benchmark_model, num_samples=N_test, return_sites=["X", "A", "Y"]
    )()
    D_test_flat = {k: v.squeeze(-1) for k, v in D_test.items()}

    model = KnownCovariateDistModel(p, link)
    conditioned_model = DataConditionedModel(model)
    guide_train = pyro.infer.autoguide.AutoDelta(conditioned_model)
    elbo = pyro.infer.Trace_ELBO()(conditioned_model, guide_train)

    # initialize parameters
    elbo(D_train)

    adam = torch.optim.Adam(elbo.parameters(), lr=0.03)

    # Do gradient steps
    for _ in range(500):
        adam.zero_grad()
        loss = elbo(D_train)
        loss.backward()
        adam.step()

    theta_hat = {
        k: v.clone().detach().requires_grad_(True) for k, v in guide_train().items()
    }
    _, analytic_eif_at_test_pts = closed_form_ate_correction(D_test_flat, theta_hat)

    mle_guide = MLEGuide(theta_hat)
    param_eif = linearize(
        PredictiveModel(model, mle_guide),
        num_samples_outer=10000,
        num_samples_inner=1,
        cg_iters=4,  # dimension of params = 4
        pointwise_influence=True,
    )

    test_data_eif = param_eif(D_test)
    median_abs_error = torch.abs(
        test_data_eif["model.guide.treatment_weight_param"] - analytic_eif_at_test_pts
    ).median()
    median_scale = torch.abs(analytic_eif_at_test_pts).median()
    if median_scale > 1:
        assert median_abs_error / median_scale < 0.5
    else:
        assert median_abs_error < 0.5

    # Test w/ pointwise_influence=False
    param_eif = linearize(
        PredictiveModel(model, mle_guide),
        num_samples_outer=10000,
        num_samples_inner=1,
        cg_iters=4,  # dimension of params = 4
        pointwise_influence=False,
    )

    test_data_eif = param_eif(D_test)
    assert torch.allclose(
        test_data_eif["model.guide.treatment_weight_param"][0],
        analytic_eif_at_test_pts.mean(),
        atol=0.5,
    )
