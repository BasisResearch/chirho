import logging
import math
import warnings
from typing import Callable, Generic, ParamSpec, TypeVar, Union

import pyro
import pytest
import scipy
import torch

import pyro.distributions as dist
import pyro.infer.reparam

pyro.settings.set(module_local_params=True)

logger = logging.getLogger(__name__)


class BaseShortColumn(pyro.nn.PyroModule):
    """
    Analytic short column deformation model
    from Section 4.3 of "OPTIMAL MODEL MANAGEMENT FOR MULTIFIDELITY MONTE CARLO ESTIMATION"
    by Benjamin Peherstorfer, Karen Willcox, and Max Gunzburger (https://epubs.siam.org/doi/abs/10.1137/15M1046472)
    """
    cost: torch.Tensor

    def __init__(self, cost: float = 1.):
        super().__init__()
        self.register_buffer("cost", torch.as_tensor(cost))
        self.register_buffer("zero", torch.tensor(0.))
        self.register_buffer("one", torch.tensor(1.))

    @pyro.nn.PyroSample
    def width(self):  # z1
        return dist.Uniform(5., 15.)

    @pyro.nn.PyroSample
    def depth(self):  # z2
        return dist.Uniform(15., 25.)

    @pyro.nn.PyroSample
    def yield_stress(self):  # z3
        return dist.LogNormal(math.log(5.), 0.5)

    @pyro.nn.PyroSample
    def bending_moment(self):  # z4
        return dist.Normal(2000., 400.)

    @pyro.nn.PyroSample
    def axial_force(self):  # z5
        return dist.Normal(500., 100.)

    @staticmethod
    def _model(width, depth, yield_stress, bending_moment, axial_force):
        raise NotImplementedError

    def forward(self) -> torch.Tensor:
        f = self._model(self.width, self.depth, self.yield_stress, self.bending_moment, self.axial_force)
        pyro.factor("log_column", torch.log(torch.abs(-f)))
        return pyro.deterministic("column", f, event_dim=0)


class HighFidelityShortColumn(BaseShortColumn):

    @staticmethod
    def _model(width, depth, yield_stress, bending_moment, axial_force):
        return 1 - 4 * bending_moment / (width * depth ** 2 * yield_stress) - \
            (axial_force / (width * depth * yield_stress)) ** 2


class LowFidelityShortColumn2(BaseShortColumn):

    @staticmethod
    def _model(width, depth, yield_stress, bending_moment, axial_force):
        return 1 - bending_moment / (width * depth ** 2 * yield_stress) - \
            (axial_force / (width * depth * yield_stress)) ** 2


class LowFidelityShortColumn3(BaseShortColumn):

    @staticmethod
    def _model(width, depth, yield_stress, bending_moment, axial_force):
        return 1 - bending_moment / (width * depth ** 2 * yield_stress) - \
            (axial_force * (1 + bending_moment) / (width * depth * yield_stress)) ** 2


class LowFidelityShortColumn4(BaseShortColumn):

    @staticmethod
    def _model(width, depth, yield_stress, bending_moment, axial_force):
        return 1 - bending_moment / (width * depth ** 2 * yield_stress) - \
            (axial_force * (1 + bending_moment) / (depth * yield_stress)) ** 2


@pytest.mark.parametrize("surrogate", [
    LowFidelityShortColumn2(cost=0.1),
    LowFidelityShortColumn3(cost=0.1),
    LowFidelityShortColumn4(cost=0.1),
    LowFidelityShortColumn2(cost=1e-5),
    LowFidelityShortColumn3(cost=1e-5),
    LowFidelityShortColumn4(cost=1e-5),
])
def test_shortcolumn_surrogates_correlation(surrogate):

    model = HighFidelityShortColumn(cost=1.)

    with pyro.plate("samples", 1e6, dim=-1), pyro.poutine.trace() as tr:
        surrogate_samples = surrogate()

    with pyro.plate("samples", 1e6, dim=-1):
        high_fidelity_samples = pyro.poutine.replay(trace=tr.trace)(model)()

    assert not torch.any(torch.isnan(high_fidelity_samples))
    assert not torch.any(torch.isnan(surrogate_samples))
    print("high fidelity mean: ", high_fidelity_samples.mean(dim=-1))
    print("surrogate mean: ", surrogate_samples.mean(dim=-1))

    print("high fidelity range: ", high_fidelity_samples.min(dim=-1)[0], high_fidelity_samples.max(dim=-1)[0])
    print("surrogate range: ", surrogate_samples.min(dim=-1)[0], surrogate_samples.max(dim=-1)[0])

    corr = scipy.stats.pearsonr(high_fidelity_samples.detach().numpy(), surrogate_samples.detach().numpy())
    print("correlation: ", corr)
    assert corr.statistic > 0.7


@pytest.mark.parametrize("surrogate", [
    LowFidelityShortColumn2(cost=0.1),
    # LowFidelityShortColumn3(cost=0.1),
    # LowFidelityShortColumn4(cost=0.1),
    # LowFidelityShortColumn2(cost=1e-5),
    # LowFidelityShortColumn3(cost=1e-5),
    # LowFidelityShortColumn4(cost=1e-5),
])
def test_multifidelity_importance_1_surrogates_mean(surrogate):

    model = HighFidelityShortColumn(cost=1.)

    # expected mean
    with pyro.plate("samples", 1e6, dim=-1):
        expected_result = model().mean(dim=-1)
 
    actual_results = []
    num_runs = 10
    for run in range(num_runs):

        # biasing distribution: draw samples from surrogate and fit a mixture model
        reparam = pyro.infer.reparam.strategies.AutoReparam(centered=0.)
        surrogate_ = reparam(surrogate)
        surrogate_guide = pyro.infer.autoguide.AutoMultivariateNormal(
            surrogate_,
            init_loc_fn=pyro.infer.autoguide.initialization.init_to_median,
            init_scale=0.1,
        )
        surrogate_elbo = pyro.infer.Trace_ELBO(num_particles=100, vectorize_particles=True)(surrogate_, surrogate_guide)
        surrogate_elbo()  # initialize parameters

        # fit surrogate guide
        with warnings.catch_warnings(), pyro.validation_enabled(False):
            warnings.filterwarnings("ignore", category=torch.jit.TracerWarning)
            # surrogate_elbo = torch.jit.trace_module(surrogate_elbo, {"forward": ()}, strict=False, check_trace=False)
            surrogate_elbo = torch.compile(surrogate_elbo, backend="inductor")
        optim = torch.optim.Adam(surrogate_elbo.parameters(), lr=0.01)
        for step in range(2000):
            loss = surrogate_elbo()
            loss.backward()
            optim.step()
            optim.zero_grad()
            if step % 100 == 0:
                print("run: ", run, "step: ", step, "loss: ", loss.item())

        # importance
        log_weights, model_trace, surrogate_guide_trace = \
            pyro.infer.importance.vectorized_importance_weights(
                pyro.poutine.block(hide=["log_column"])(reparam(model)),
                surrogate_guide,
                num_samples=1e6,
                max_plate_nesting=0,
            )

        result = torch.mean(log_weights.exp() * model_trace.nodes["column"]["value"], dim=-1)
        print(result, torch.logsumexp(log_weights - log_weights.shape[-1], dim=-1).exp())
        assert not torch.any(torch.isnan(result))
        if not torch.isnan(result).any():
            actual_results.append(result.item())

    print("expected high-fidelity mean: ", expected_result.item())
    print("estimated multifidelity mean: ", torch.tensor(actual_results).mean().item())
    print("mse: ", torch.mean((expected_result - torch.tensor(actual_results)) ** 2))


def test_multifidelity_importance_n_surrogates():

    high_fidelity = HighFidelityShortColumn(cost=1.)

    surrogates = [
        LowFidelityShortColumn2(cost=0.1),
        LowFidelityShortColumn3(cost=0.1),
        LowFidelityShortColumn4(cost=0.1),
        LowFidelityShortColumn2(cost=1e-5),
        LowFidelityShortColumn3(cost=1e-5),
        LowFidelityShortColumn4(cost=1e-5),
    ]

    surrogate_guides = []
    for surrogate in surrogates:
        surrogate_guide = pyro.infer.autoguide.AutoMultivariateNormal(surrogate)
        surrogate_elbo = pyro.infer.Trace_ELBO()(surrogate, surrogate_guide)

        # fit surrogate guide
        surrogate_elbo()  # initialize parameters
        optim = torch.optim.Adam(surrogate_elbo.parameters(), lr=0.01)
        for step in range(1000):
            loss = surrogate_elbo()
            loss.backward()
            optim.step()
            optim.zero_grad()

        # append trained surrogate guide
        surrogate_guides.append(surrogate_guide)

    # mixture weights for multifidelity biasing distribution
    mixture_weights = torch.tensor([
        1 / surrogate.cost - pyro.infer.Trace_ELBO(num_particles=100)(high_fidelity, surrogate_guide)()
        for surrogate, surrogate_guide in zip(surrogates, surrogate_guides)
    ])

    # multifidelity biasing distribution
    def multifidelity_biasing_distribution():
        index = pyro.sample("index", dist.Categorical(probs=mixture_weights))
        return surrogate_guides[index]()

    # importance weights
    log_weights, model_trace, surrogate_guide_trace = \
        pyro.infer.importance.vectorized_importance_weights(
            high_fidelity, multifidelity_biasing_distribution, num_samples=1000, max_plate_nesting=1
        )
    result = torch.logsumexp(log_weights, dim=-1).exp()
    assert not torch.any(torch.isnan(result))
    print(result)
