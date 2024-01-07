import functools
import warnings

import pyro
import pytest
import torch

from chirho.indexed.handlers import IndexPlatesMessenger
from chirho.indexed.ops import indices_of
from chirho.robust.internals.linearize import (
    conjugate_gradient_solve,
    make_empirical_fisher_vp,
)
from chirho.robust.internals.predictive import (
    BatchedLatents,
    BatchedNMCLogPredictiveLikelihood,
    BatchedObservations,
    KernelPerturbedModel,
    PredictiveFunctional,
    PredictiveModel
)
from chirho.robust.internals.utils import make_functional_call, reset_rng_state

from .robust_fixtures import SimpleGuide, SimpleModel

pyro.settings.set(module_local_params=True)


def test_empirical_fisher_vp_nmclikelihood_cg_composition():
    model = SimpleModel()
    guide = SimpleGuide()
    model(), guide()  # initialize
    log_prob = BatchedNMCLogPredictiveLikelihood(model, guide, num_samples=100)
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
        make_empirical_fisher_vp(func_log_prob, log_prob_params, data)
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
    assert all(fvp_vk.shape == v[k].shape for k, fvp_vk in fvp(v).items())

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

    log_prob = BatchedNMCLogPredictiveLikelihood(
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


@pytest.mark.parametrize("pad_dim", [0, 1, 2])
def test_batched_observations(pad_dim: int):
    max_plate_nesting = 1 + pad_dim
    obs_plate_name = "__dummy_plate__"
    num_particles_obs = 3
    model = SimpleModel()
    guide = SimpleGuide()

    model(), guide()  # initialize

    predictive = pyro.infer.Predictive(
        model, num_samples=num_particles_obs, return_sites=["y"]
    )

    test_data = predictive()

    with IndexPlatesMessenger(first_available_dim=-max_plate_nesting - 1):
        with pyro.poutine.trace() as tr:
            with BatchedObservations(test_data, name=obs_plate_name):
                model()

        tr.trace.compute_log_prob()

        for name, node in tr.trace.nodes.items():
            if node["type"] == "sample" and not pyro.poutine.util.site_is_subsample(
                node
            ):
                if name in test_data:
                    assert obs_plate_name in indices_of(node["log_prob"], event_dim=0)
                    assert obs_plate_name in indices_of(
                        node["value"], event_dim=len(node["fn"].event_shape)
                    )
                else:
                    assert obs_plate_name not in indices_of(
                        node["log_prob"], event_dim=0
                    )
                    assert obs_plate_name not in indices_of(
                        node["value"], event_dim=len(node["fn"].event_shape)
                    )


@pytest.mark.parametrize("pad_dim", [0, 1, 2])
def test_batched_latents_observations(pad_dim: int):
    max_plate_nesting = 1 + pad_dim
    num_particles_latent = 5
    num_particles_obs = 3
    obs_plate_name = "__dummy_plate__"
    latent_plate_name = "__dummy_latents__"
    model = SimpleModel()
    guide = SimpleGuide()

    model(), guide()  # initialize

    predictive = pyro.infer.Predictive(
        model, num_samples=num_particles_obs, return_sites=["y"]
    )

    test_data = predictive()

    with IndexPlatesMessenger(first_available_dim=-max_plate_nesting - 1):
        with pyro.poutine.trace() as tr:
            with BatchedLatents(
                num_particles=num_particles_latent, name=latent_plate_name
            ):
                with BatchedObservations(test_data, name=obs_plate_name):
                    model()

        tr.trace.compute_log_prob()

        for name, node in tr.trace.nodes.items():
            if node["type"] == "sample" and not pyro.poutine.util.site_is_subsample(
                node
            ):
                if name in test_data:
                    assert obs_plate_name in indices_of(node["log_prob"], event_dim=0)
                    assert obs_plate_name in indices_of(
                        node["value"], event_dim=len(node["fn"].event_shape)
                    )
                    assert latent_plate_name in indices_of(
                        node["log_prob"], event_dim=0
                    )
                    assert latent_plate_name not in indices_of(
                        node["value"], event_dim=len(node["fn"].event_shape)
                    )
                else:
                    assert latent_plate_name in indices_of(
                        node["log_prob"], event_dim=0
                    )
                    assert latent_plate_name in indices_of(
                        node["value"], event_dim=len(node["fn"].event_shape)
                    )


def test_kernel_perturbation():
    model = SimpleModel()
    guide = SimpleGuide()

    model(), guide()  # initialize

    predictive = pyro.infer.Predictive(
        model, num_samples=1, return_sites=["y"]
    )

    # Unwrap just for this test that operates independently of plating/batching.
    test_data = dict(y=predictive()['y'][0])

    with pyro.poutine.trace() as model_tr_messenger:
        model()

    kpmodel1 = KernelPerturbedModel(model, eps=torch.tensor(1.0))
    with pyro.poutine.trace() as kpmodel1_tr_messenger:
        kpmodel1(_kernel_loc=test_data)

    kpmodel2 = KernelPerturbedModel(model, eps=torch.tensor(0.0))
    with pyro.poutine.trace() as kpmodel2_tr_messenger:
        kpmodel2(_kernel_loc=test_data)

    pred_kpmodel1 = PredictiveModel(kpmodel1, guide)
    with pyro.poutine.trace() as pred_kpmodel1_tr_messenger:
        pred_kpmodel1(_kernel_loc=test_data)

    pred_kpmodel2 = PredictiveModel(kpmodel2, guide)
    with pyro.poutine.trace() as pred_kpmodel2_tr_messenger:
        pred_kpmodel2(_kernel_loc=test_data)

    # Require that kpmodel1, pred_kpmodel1, and test data give the same values for y.
    assert torch.allclose(
        kpmodel1_tr_messenger.trace.nodes["y"]["value"],
        test_data["y"],
    )
    assert torch.allclose(
        pred_kpmodel1_tr_messenger.trace.nodes["y"]["value"],
        test_data["y"],
    )

    # Require that model, kpmodel2, pred_kpmodel2, do not give the test data. Check for very high precision match,
    #  as we want there to basically zero probability of a false equality due to random sample from the model.
    assert not torch.allclose(
        model_tr_messenger.trace.nodes["y"]["value"],
        test_data["y"],
        atol=1e-9,
    )
    assert not torch.allclose(
        kpmodel2_tr_messenger.trace.nodes["y"]["value"],
        test_data["y"],
        atol=1e-9,
    )
    assert not torch.allclose(
        pred_kpmodel2_tr_messenger.trace.nodes["y"]["value"],
        test_data["y"],
        atol=1e-9,
    )


@pytest.mark.parametrize("pad_dim", [0, 1, 2])
def test_kernel_perturbation_composition_batched(pad_dim: int):
    """
    This emulates test_kernel_perturbation but with batching.
    """
    max_plate_nesting = 1 + pad_dim
    num_particles_latent = 5
    num_particles_obs = 3
    obs_plate_name = "__dummy_plate__"
    latent_plate_name = "__dummy_latents__"
    model = SimpleModel()
    guide = SimpleGuide()

    model(), guide()  # initialize

    predictive = pyro.infer.Predictive(
        model, num_samples=num_particles_obs, return_sites=["y"]
    )

    test_data = predictive()

    def get_trace(_model, incl_kernel_loc=True):
        with IndexPlatesMessenger(first_available_dim=-max_plate_nesting - 1):
            with pyro.poutine.trace() as tr:
                with BatchedLatents(num_particles=num_particles_latent, name=latent_plate_name):
                    with BatchedObservations(test_data, name=obs_plate_name):
                        if incl_kernel_loc:
                            _model(_kernel_loc=test_data)
                        else:
                            _model()
        return tr.trace

    model_trace = get_trace(model, incl_kernel_loc=False)

    kpmodel1 = KernelPerturbedModel(model, eps=torch.tensor(1.0))
    kpmodel1_trace = get_trace(kpmodel1)

    kpmodel2 = KernelPerturbedModel(model, eps=torch.tensor(0.0))
    kpmodel2_trace = get_trace(kpmodel2)

    # test failing see FIXME 2880dhsl in robust/internals/predictive.py

    # TODO go through the predictive models like the other test and assert equality/non-equality to make sure
    #  things are coming from the right distribution.





