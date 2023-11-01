import logging
from typing import Callable, Generic, ParamSpec, TypeVar, Union

import pyro
import torch

import pyro.distributions as dist

pyro.settings.set(module_local_params=True)

logger = logging.getLogger(__name__)


class BaseShortColumn(pyro.nn.PyroModule):
    cost: torch.Tensor

    def __init__(self, cost: float = 1.):
        super().__init__()
        self.register_buffer("cost", torch.as_tensor(cost))

    @pyro.nn.PyroSample
    def width(self):  # z1
        return dist.Uniform(5., 15.)

    @pyro.nn.PyroSample
    def depth(self):  # z2
        return dist.Uniform(15., 25.)

    @pyro.nn.PyroSample
    def yield_stress(self):  # z3
        return dist.LogNormal(5., 0.5)

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
        pyro.factor("column", torch.log(f))
        return f


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


def test_multifidelity_importance_2_surrogates():

    model = HighFidelityShortColumn(cost=1.)
    surrogate = LowFidelityShortColumn2(cost=0.1)

    # biasing distribution: draw samples from surrogate and fit a mixture model
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

    # importance weights
    log_weights, model_trace, surrogate_guide_trace = \
        pyro.infer.importance.vectorized_importance_weights(model, surrogate_guide, num_samples=1000, max_plate_nesting=1)
    result = torch.logsumexp(log_weights, dim=-1).exp()
    assert not torch.any(torch.isnan(result))
    print(result)


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
