import functools
import warnings
from typing import Callable, List, Mapping, Optional, Set, Tuple, TypeVar

import pyro
import pytest
import torch
from typing_extensions import ParamSpec

from chirho.robust.handlers.predictive import PredictiveFunctional, PredictiveModel
from chirho.robust.ops import influence_fn

from .robust_fixtures import SimpleGuide, SimpleModel

pyro.settings.set(module_local_params=True)

P = ParamSpec("P")
Q = ParamSpec("Q")
S = TypeVar("S")
T = TypeVar("T")


ModelTestCase = Tuple[
    Callable[[], Callable], Callable[[Callable], Callable], Set[str], Optional[int]
]

MODEL_TEST_CASES: List[ModelTestCase] = [
    (SimpleModel, lambda _: SimpleGuide(), {"y"}, 1),
    (SimpleModel, lambda _: SimpleGuide(), {"y"}, None),
    pytest.param(
        SimpleModel,
        lambda m: pyro.infer.autoguide.AutoNormal(pyro.poutine.block(hide=["y"])(m)),
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
@pytest.mark.parametrize("num_predictive_samples", [1, 5])
def test_nmc_predictive_influence_smoke(
    model,
    guide,
    obs_names,
    max_plate_nesting,
    num_samples_outer,
    num_samples_inner,
    cg_iters,
    num_predictive_samples,
):
    model = model()
    guide = guide(model)
    model(), guide()  # initialize

    with torch.no_grad():
        test_datum = {
            k: v[0]
            for k, v in pyro.infer.Predictive(
                model, num_samples=2, return_sites=obs_names, parallel=True
            )().items()
        }

    predictive_eif = influence_fn(
        functools.partial(PredictiveFunctional, num_samples=num_predictive_samples),
        test_datum,
        max_plate_nesting=max_plate_nesting,
        num_samples_outer=num_samples_outer,
        num_samples_inner=num_samples_inner,
        cg_iters=cg_iters,
    )(PredictiveModel(model, guide))

    test_datum_eif: Mapping[str, torch.Tensor] = predictive_eif()
    assert len(test_datum_eif) > 0
    for k, v in test_datum_eif.items():
        assert not torch.isnan(v).any(), f"eif for {k} had nans"
        assert not torch.isinf(v).any(), f"eif for {k} had infs"
        assert not torch.isclose(v, torch.zeros_like(v)).all(), f"eif for {k} was zero"


@pytest.mark.parametrize("model,guide,obs_names,max_plate_nesting", MODEL_TEST_CASES)
@pytest.mark.parametrize("num_samples_outer,num_samples_inner", [(10, None), (10, 100)])
@pytest.mark.parametrize("cg_iters", [None, 1, 10])
@pytest.mark.parametrize("num_predictive_samples", [1, 5])
def test_nmc_predictive_influence_vmap_smoke(
    model,
    guide,
    obs_names,
    max_plate_nesting,
    num_samples_outer,
    num_samples_inner,
    cg_iters,
    num_predictive_samples,
):
    model = model()
    guide = guide(model)

    model(), guide()  # initialize

    with torch.no_grad():
        test_data = pyro.infer.Predictive(
            model, num_samples=4, return_sites=obs_names, parallel=True
        )()

    predictive_eif = influence_fn(
        functools.partial(PredictiveFunctional, num_samples=num_predictive_samples),
        test_data,
        max_plate_nesting=max_plate_nesting,
        num_samples_outer=num_samples_outer,
        num_samples_inner=num_samples_inner,
        cg_iters=cg_iters,
    )(PredictiveModel(model, guide))

    test_data_eif: Mapping[str, torch.Tensor] = predictive_eif()
    assert len(test_data_eif) > 0
    for k, v in test_data_eif.items():
        assert not torch.isnan(v).any(), f"eif for {k} had nans"
        assert not torch.isinf(v).any(), f"eif for {k} had infs"
        assert not torch.isclose(v, torch.zeros_like(v)).all(), f"eif for {k} was zero"


def test_influence_raises_no_grad_warning_correctly():
    model = SimpleModel()
    guide = SimpleGuide()
    predictive = pyro.infer.Predictive(
        model, guide=guide, num_samples=10, return_sites=["y"]
    )
    points = predictive()
    influence = influence_fn(
        PredictiveFunctional,
        points,
        num_samples_outer=10,
        num_samples_inner=10,
    )(PredictiveModel(model, guide))

    with pytest.warns(UserWarning, match="torch.no_grad"):
        influence()

    with pytest.warns() as record:
        with torch.no_grad():
            influence()
        assert len(record) == 0
        warnings.warn("Dummy warning.")
