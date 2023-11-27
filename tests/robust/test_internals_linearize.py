import functools
from typing import Callable, List, Mapping, Optional, Set, Tuple, TypeVar

import pyro
import pyro.distributions as dist
import pytest
import torch
from typing_extensions import ParamSpec

from chirho.robust.internals.linearize import conjugate_gradient_solve, linearize

pyro.settings.set(module_local_params=True)

P = ParamSpec("P")
Q = ParamSpec("Q")
S = TypeVar("S")
T = TypeVar("T")


@pytest.mark.parametrize("ndim", [1, 2, 3, 10])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_cg_solve(ndim: int, dtype: torch.dtype):
    cg_iters = None
    residual_tol = 1e-10

    A = torch.eye(ndim, dtype=dtype) + 0.1 * torch.rand(ndim, ndim, dtype=dtype)
    expected_x = torch.randn(ndim, dtype=dtype)
    b = A @ expected_x

    actual_x = conjugate_gradient_solve(
        lambda v: A @ v, b, cg_iters=cg_iters, residual_tol=residual_tol
    )
    assert torch.sum((actual_x - expected_x) ** 2) < 1e-4


@pytest.mark.parametrize("ndim", [1, 2, 3, 10])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("num_particles", [1, 4])
def test_batch_cg_solve(ndim: int, dtype: torch.dtype, num_particles: int):
    cg_iters = None
    residual_tol = 1e-10

    A = torch.eye(ndim, dtype=dtype) + 0.1 * torch.rand(ndim, ndim, dtype=dtype)
    expected_x = torch.randn(num_particles, ndim, dtype=dtype)
    b = torch.einsum("ij,nj->ni", A, expected_x)
    assert b.shape == (num_particles, ndim)

    batch_solve = torch.vmap(
        functools.partial(
            conjugate_gradient_solve,
            lambda v: A @ v,
            cg_iters=cg_iters,
            residual_tol=residual_tol,
        ),
    )
    actual_x = batch_solve(b)

    assert torch.all(torch.sum((actual_x - expected_x) ** 2, dim=1) < 1e-4)


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
        model,
        guide,
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
        assert not torch.isclose(v, torch.zeros_like(v)).all(), f"eif for {k} was zero"


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
        model,
        guide,
        max_plate_nesting=max_plate_nesting,
        num_samples_outer=num_samples_outer,
        num_samples_inner=num_samples_inner,
        cg_iters=cg_iters,
    )

    with torch.no_grad():
        test_data = pyro.infer.Predictive(
            model, num_samples=4, return_sites=obs_names, parallel=True
        )()

    batch_param_eif = torch.vmap(param_eif, randomness="different")
    test_data_eif: Mapping[str, torch.Tensor] = batch_param_eif(test_data)
    assert len(test_data_eif) > 0
    for k, v in test_data_eif.items():
        assert not torch.isnan(v).any(), f"eif for {k} had nans"
        assert not torch.isinf(v).any(), f"eif for {k} had infs"
        assert not torch.isclose(v, torch.zeros_like(v)).all(), f"eif for {k} was zero"
