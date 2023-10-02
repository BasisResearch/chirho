import logging

import pyro
import pytest
import torch
from pyro.distributions import Normal

from chirho.counterfactual.handlers import TwinWorldCounterfactual
from chirho.dynamical.handlers import (
    DynamicTrace,
    NonInterruptingPointObservationArray,
    SimulatorEventLoop,
    StaticIntervention,
)
from chirho.dynamical.handlers.ODE.solvers import TorchDiffEq
from chirho.dynamical.ops import State, simulate
from chirho.observational.handlers.soft_conditioning import AutoSoftConditioning
from tests.dynamical.dynamical_fixtures import (
    UnifiedFixtureDynamics,
    run_svi_inference_torch_direct,
)

logger = logging.getLogger(__name__)

# Global variables for tests
init_state = State(S=torch.tensor(10.0), I=torch.tensor(1.0), R=torch.tensor(0.0))
start_time = torch.tensor(0.0)
end_time = torch.tensor(1.2)
logging_times = torch.tensor([0.1, 0.4, 0.8])

#
# 15 passengers tested positive for a disease after landing
landing_data = {"infected_passengers": torch.tensor(15.0)}
landing_time = 0.3 + 1e-2

# In the counterfactual world, a super-spreader event occured at time 0.1
# In the factual world, however, this did not occur.

ssd = torch.tensor(2.0)

counterfactual = State(
    S=lambda s: s - ssd,
    I=lambda i: i + ssd,
)
superspreader_time = 0.1

# We want to know how many people passengers would have bene infected at the
#  time of landing had the super-spreader event not occurred.

flight_landing_times = torch.tensor(
    [landing_time, landing_time + 1e-2, landing_time + 2e-2]
)
flight_landing_data = {k: torch.tensor([v] * 3) for (k, v) in landing_data.items()}
reparam_config = AutoSoftConditioning(scale=0.01, alpha=0.5)

twin_world = TwinWorldCounterfactual()
intervention = StaticIntervention(time=superspreader_time, intervention=counterfactual)
reparam = pyro.poutine.reparam(config=reparam_config)
vec_obs3 = NonInterruptingPointObservationArray(
    times=flight_landing_times, data=flight_landing_data
)


def counterf_model():
    with SimulatorEventLoop():
        with vec_obs3, reparam, twin_world, intervention:
            return simulate(
                UnifiedFixtureDynamicsReparam(beta=0.5, gamma=0.7),
                init_state,
                start_time,
                end_time,
                solver=TorchDiffEq(),
            )


def conditioned_model():
    # This is equivalent to the following:
    # with SimulatorEventLoop():
    #   with vec_obs3:
    #       return simulate(...)
    # It simply blocks the intervention, twin world, and reparameterization handlers, as those need to be removed from
    #  the factual conditional world.
    with pyro.poutine.messenger.block_messengers(
        lambda m: m in (reparam, twin_world, intervention)
    ):
        return counterf_model()


# A reparameterized observation function of various flight arrivals.
class UnifiedFixtureDynamicsReparam(UnifiedFixtureDynamics):
    def observation(self, X: State[torch.Tensor]):
        # super().observation(X)

        # A flight arrives in a country that tests all arrivals for a disease. The number of people infected on the
        #  plane is a noisy function of the number of infected people in the country of origin at that time.
        u_ip = pyro.sample("u_ip", Normal(7.0, 2.0).expand(X.I.shape[-1:]).to_event(1))
        pyro.deterministic("infected_passengers", X.I + u_ip, event_dim=1)


def test_shape_twincounterfactual_observation_intervention_commutes():
    with DynamicTrace(logging_times) as dt:
        with pyro.poutine.trace() as tr:
            conditioned_model()

    ret = dt.trace

    num_worlds = 2

    state_shape = (num_worlds, len(logging_times))
    assert ret.S.squeeze().squeeze().shape == state_shape
    assert ret.I.squeeze().squeeze().shape == state_shape
    assert ret.R.squeeze().squeeze().shape == state_shape

    nodes = tr.get_trace().nodes

    obs_shape = (num_worlds, len(flight_landing_times))
    assert nodes["infected_passengers"]["value"].squeeze().shape == obs_shape


# TODO: This test is failing because the autoguide doesn't recognize any latents in the model.
@pytest.mark.skip
def test_smoke_inference_twincounterfactual_observation_intervention_commutes():
    # Run inference on factual model.
    guide = run_svi_inference_torch_direct(conditioned_model, n_steps=2, verbose=False)

    num_samples = 100
    pred = pyro.infer.Predictive(counterf_model, guide=guide, num_samples=num_samples)()

    num_worlds = 2
    # infected passengers is going to differ depending on which of two worlds
    assert pred["infected_passengers"].squeeze().shape == (
        num_samples,
        num_worlds,
        len(flight_landing_times),
    )
    # Noise is shared between factual and counterfactual worlds.
    assert pred["u_ip"].squeeze().shape == (num_samples, len(flight_landing_times))


@pytest.mark.skip
def test_shape_multicounterfactual_observation_intervention_commutes():
    raise NotImplementedError()
