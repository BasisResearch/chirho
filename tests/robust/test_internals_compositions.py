import functools
import warnings

import pyro
import pytest
import torch

from chirho.robust.internals.linearize import (
    conjugate_gradient_solve,
    make_empirical_fisher_vp,
)
from chirho.robust.internals.predictive import (
    NMCLogPredictiveLikelihood,
    PointLogPredictiveLikelihood,
)
from chirho.robust.internals.utils import make_functional_call, reset_rng_state

from .robust_fixtures import SimpleGuide, SimpleModel

pyro.settings.set(module_local_params=True)

prob_config = [
    (NMCLogPredictiveLikelihood, False),
    (PointLogPredictiveLikelihood, True),
]


@pytest.mark.parametrize("prob_config", prob_config)
def test_empirical_fisher_vp_nmclikelihood_cg_composition(prob_config):
    prob_fn, is_batched = prob_config
    model = SimpleModel()
    guide = SimpleGuide()
    model(), guide()  # initialize
    log_prob = prob_fn(model, guide)
    log_prob_params, func_log_prob = make_functional_call(log_prob)
    func_log_prob = reset_rng_state(pyro.util.get_rng_state())(func_log_prob)

    predictive = pyro.infer.Predictive(
        model, guide=guide, num_samples=1000, parallel=True, return_sites=["y"]
    )
    predictive_params, func_predictive = make_functional_call(predictive)

    cg_solver = functools.partial(conjugate_gradient_solve, cg_iters=2)

    with torch.no_grad():
        data = func_predictive(predictive_params)
    fvp = torch.func.vmap(
        make_empirical_fisher_vp(
            func_log_prob, log_prob_params, data, is_batched=is_batched
        )
    )

    v = {
        k: torch.ones_like(v).unsqueeze(0)
        if k != "guide.loc_a"
        else torch.zeros_like(v).unsqueeze(0)
        for k, v in log_prob_params.items()
    }

    # For this model, fvp for loc_a is zero. See
    # https://github.com/BasisResearch/chirho/issues/427
    assert fvp(v)["guide.loc_a"].abs().max() == 0

    solve_one = cg_solver(fvp, v)
    solve_two = cg_solver(fvp, v)

    if solve_one["guide.loc_a"].abs().max() > 1e6:
        warnings.warn(
            "solve_one['guide.loc_a'] is large (max entry={}).".format(
                solve_one["guide.loc_a"].abs().max()
            )
        )

    if solve_one["guide.loc_b"].abs().max() > 1e6:
        warnings.warn(
            "solve_one['guide.loc_b'] is large (max entry={}).".format(
                solve_one["guide.loc_b"].abs().max()
            )
        )

    assert torch.allclose(
        solve_one["guide.loc_a"], torch.zeros_like(log_prob_params["guide.loc_a"])
    )
    assert torch.allclose(solve_one["guide.loc_a"], solve_two["guide.loc_a"])
    assert torch.allclose(solve_one["guide.loc_b"], solve_two["guide.loc_b"])


link_functions = [
    lambda mu: pyro.distributions.Normal(mu, 1.0),
    lambda mu: pyro.distributions.Bernoulli(logits=mu),
    lambda mu: pyro.distributions.Beta(concentration1=mu, concentration0=1.0),
    lambda mu: pyro.distributions.Exponential(rate=mu),
]


@pytest.mark.parametrize("link_fn", link_functions)
def test_nmc_likelihood_seeded(link_fn):
    model = SimpleModel(link_fn=link_fn)
    guide = SimpleGuide()
    model(), guide()  # initialize

    log_prob = NMCLogPredictiveLikelihood(
        model, guide, num_samples=3, max_plate_nesting=3
    )
    log_prob_params, func_log_prob = make_functional_call(log_prob)

    func_log_prob = reset_rng_state(pyro.util.get_rng_state())(func_log_prob)

    datapoint = {"y": torch.tensor([1.0, 2.0, 3.0])}
    prob_call_one = func_log_prob(log_prob_params, datapoint)
    prob_call_two = func_log_prob(log_prob_params, datapoint)
    prob_call_three = func_log_prob(log_prob_params, datapoint)
    assert torch.allclose(prob_call_two, prob_call_three)
    assert torch.allclose(prob_call_one, prob_call_two)

    data = {"y": torch.tensor([[0.3665, 1.5440, 2.2210], [0.3665, 1.5440, 2.2210]])}

    fvp = make_empirical_fisher_vp(func_log_prob, log_prob_params, data)

    v = {k: torch.ones_like(v) for k, v in log_prob_params.items()}

    assert (fvp(v)["guide.loc_a"].abs().max() + fvp(v)["guide.loc_b"].abs().max()) > 0

    # Check if fvp agrees across multiple calls of same `fvp` object
    assert torch.allclose(fvp(v)["guide.loc_a"], fvp(v)["guide.loc_a"])
    assert torch.allclose(fvp(v)["guide.loc_b"], fvp(v)["guide.loc_b"])
