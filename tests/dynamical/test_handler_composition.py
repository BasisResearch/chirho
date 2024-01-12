import logging

import pyro
import pytest
import torch
from pyro.distributions import Normal

from chirho.counterfactual.handlers import TwinWorldCounterfactual
from chirho.dynamical.handlers import (
    LogTrajectory,
    StaticBatchObservation,
    StaticIntervention,
)
from chirho.dynamical.handlers.solver import TorchDiffEq
from chirho.dynamical.internals._utils import ShallowMessenger
from chirho.dynamical.ops import State, simulate
from chirho.observational.handlers import condition
from chirho.observational.handlers.soft_conditioning import AutoSoftConditioning
from tests.dynamical.dynamical_fixtures import (
    UnifiedFixtureDynamics,
    run_svi_inference_torch_direct,
)

logger = logging.getLogger(__name__)

# Global variables for tests
init_state = dict(S=torch.tensor(10.0), I=torch.tensor(1.0), R=torch.tensor(0.0))
start_time = torch.tensor(0.0)
end_time = torch.tensor(1.2)
logging_times = torch.tensor([0.3, 0.6, 0.9])

#
# 15 passengers tested positive for a disease after landing
landing_data = {"infected_passengers": torch.tensor(15.0)}
landing_time = 0.3 + 1e-2

# In the counterfactual world, a super-spreader event occured at time 0.1
# In the factual world, however, this did not occur.

ssd = torch.tensor(2.0)

counterfactual = dict(
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


def counterf_model():
    model = UnifiedFixtureDynamicsReparam(beta=0.5, gamma=0.7)
    obs = condition(data=flight_landing_data)(model.observation)
    vec_obs3 = StaticBatchObservation(times=flight_landing_times, observation=obs)
    with vec_obs3:
        with TorchDiffEq():
            with reparam, twin_world, intervention:
                return simulate(
                    model,
                    init_state,
                    start_time,
                    end_time,
                )


def conditioned_model():
    # This is equivalent to the following:
    # with Solver():
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
        u_ip = pyro.sample(
            "u_ip", Normal(7.0, 2.0).expand(X["I"].shape[-1:]).to_event(1)
        )
        pyro.deterministic("infected_passengers", X["I"] + u_ip, event_dim=1)


def test_shape_twincounterfactual_observation_intervention_commutes():
    with LogTrajectory(logging_times) as dt:
        with pyro.poutine.trace() as tr:
            conditioned_model()

    ret = dt.trajectory

    num_worlds = 2

    state_shape = (num_worlds, len(logging_times))
    assert ret["S"].squeeze().squeeze().shape == state_shape
    assert ret["I"].squeeze().squeeze().shape == state_shape
    assert ret["R"].squeeze().squeeze().shape == state_shape

    nodes = tr.get_trace().nodes

    obs_shape = (num_worlds, len(flight_landing_times))
    assert nodes["infected_passengers"]["value"].squeeze().shape == obs_shape


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


class ShallowLogSample(ShallowMessenger):
    log: set

    def __enter__(self):
        self.log = set()
        return super().__enter__()

    def _pyro_sample(self, msg):
        self.log.add(msg["name"])
        pyro.sample(f"{msg['name']}__2", pyro.distributions.Bernoulli(0.5))


def test_shallow_handler_stack():
    with pyro.poutine.trace() as tr1:
        with ShallowLogSample() as h1, ShallowLogSample() as h2:
            pyro.sample("a", pyro.distributions.Bernoulli(0.5))
            with pyro.poutine.trace() as tr2, ShallowLogSample() as h3:
                pyro.sample("b", pyro.distributions.Bernoulli(0.5))

    assert h1.log == {"a__2"}
    assert h2.log == {"a"}
    assert h3.log == {"b"}

    assert set(tr1.trace.nodes.keys()) == {"a", "a__2", "a__2__2", "b", "b__2"}
    assert set(tr2.trace.nodes.keys()) == {"b", "b__2"}


def test_shallow_handler_block():
    with ShallowLogSample() as h1, pyro.poutine.block(hide_types=["sample"]):
        pyro.sample("a", pyro.distributions.Bernoulli(0.5))
        with ShallowLogSample() as h2:
            pyro.sample("b", pyro.distributions.Bernoulli(0.5))

    assert h1.log == set()
    assert h2.log == {"b"}


@pytest.mark.parametrize("error_before_sample", [True, False])
def test_shallow_handler_error(error_before_sample):
    class ShallowLogSampleError(ShallowLogSample):
        def _pyro_sample(self, msg):
            super()._pyro_sample(msg)
            raise RuntimeError("error")

    with pytest.raises(RuntimeError, match="error"):
        with ShallowLogSampleError():
            if error_before_sample:
                raise RuntimeError("error")
            pyro.sample("a", pyro.distributions.Bernoulli(0.5))

    assert not pyro.poutine.runtime._PYRO_STACK
