import functools
from typing import Callable, List, Mapping, Optional, Set, Tuple, TypeVar

import pyro
import pytest
import torch
from typing_extensions import ParamSpec

from chirho.robust.handlers.estimators import one_step_correction, tmle
from chirho.robust.internals.predictive import PredictiveFunctional, PredictiveModel

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
@pytest.mark.parametrize("num_predictive_samples", [1, 5])
def test_one_step_correction_smoke(
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

    one_step = tmle(
        model,
        guide,
        functional=functools.partial(
            PredictiveFunctional, num_samples=num_predictive_samples
        ),
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

    one_step_on_test: Mapping[str, torch.Tensor] = one_step(test_datum)
    assert len(one_step_on_test) > 0
    for k, v in one_step_on_test.items():
        assert not torch.isnan(v).any(), f"one_step for {k} had nans"
        assert not torch.isinf(v).any(), f"one_step for {k} had infs"
        assert not torch.isclose(
            v, torch.zeros_like(v)
        ).all(), f"one_step for {k} was zero"
