import copy
import functools
from typing import Callable, List, Mapping, Optional, Set, Tuple, TypeVar

import pyro
import pytest
import torch
from typing_extensions import ParamSpec

from chirho.robust.handlers.estimators import one_step_corrected_estimator, tmle
from chirho.robust.handlers.predictive import PredictiveFunctional, PredictiveModel

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
@pytest.mark.parametrize("estimation_method", [one_step_corrected_estimator, tmle])
def test_estimator_smoke(
    model,
    guide,
    obs_names,
    max_plate_nesting,
    num_samples_outer,
    num_samples_inner,
    cg_iters,
    num_predictive_samples,
    estimation_method,
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

    predictive_model = PredictiveModel(model, guide)

    prev_params = copy.deepcopy(dict(predictive_model.named_parameters()))

    if estimation_method == tmle:
        estimator_kwargs = {
            "n_tmle_steps": 1,
            "n_grad_steps": 2,
            "num_nmc_samples": 10,
            "num_grad_samples": 10,
        }
    else:
        estimator_kwargs = {}

    estimator = estimation_method(
        functools.partial(PredictiveFunctional, num_samples=num_predictive_samples),
        test_datum,
        max_plate_nesting=max_plate_nesting,
        num_samples_outer=num_samples_outer,
        num_samples_inner=num_samples_inner,
        cg_iters=cg_iters,
        **estimator_kwargs,
    )(predictive_model)

    estimate_on_test: Mapping[str, torch.Tensor] = estimator()
    assert len(estimate_on_test) > 0
    for k, v in estimate_on_test.items():
        assert not torch.isnan(v).any(), f"{estimation_method} for {k} had nans"
        assert not torch.isinf(v).any(), f"{estimation_method} for {k} had infs"
        assert not torch.isclose(
            v, torch.zeros_like(v)
        ).all(), f"{estimation_method} estimator for {k} was zero"

    # Assert estimator doesn't have side effects on model parameters.
    new_params = dict(predictive_model.named_parameters())
    for k, v in prev_params.items():
        assert torch.allclose(v, new_params[k]), f"{k} was updated"
