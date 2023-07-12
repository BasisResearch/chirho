import logging
from contextlib import ExitStack

import pyro
import pytest
import torch
from pyro.infer import SVI, Trace_ELBO
from pyro.infer.autoguide import AutoMultivariateNormal

from chirho.dynamical.handlers import (
    NonInterruptingPointObservationArray,
    PointObservation,
    SimulatorEventLoop,
    simulate,
)
from chirho.dynamical.ops import State

from chirho.counterfactual.handlers import TwinWorldCounterfactual

from .dynamical_fixtures import (
    SimpleSIRDynamics,
    bayes_sir_model,
    check_trajectories_match,
)

from chirho.dynamical.handlers import ODEDynamics, PointIntervention
from chirho.dynamical.ops import State, Trajectory

from pyro.distributions import Normal, Poisson, Uniform, constraints

logger = logging.getLogger(__name__)

# Global variables for tests
init_state = State(S=torch.tensor(10.0), I=torch.tensor(1.0), R=torch.tensor(0.0))
tspan = torch.tensor([0.0, .3, .6, .9, 1.2])


class SimpleSIRDynamicsBayesReparam(SimpleSIRDynamics):

    def observation(self, X: State[torch.Tensor]):
        obs = super().observation(X)

        # A flight arrives in a country that tests all arrivals for a disease. The number of people infected on the
        #  plane is a noisy function of the number of infected people in the country of origin at that time.
        u_ip = pyro.sample("u_ip", Poisson(4))
        pyro.deterministic("infected_passengers", obs["I_obs"] + u_ip)


def test_condition_counterfactual_commutes():

    # Four passengers tested positive for a disease after landing
    landing_data = {"infected_passengers": torch.tensor(4.0)}
    landing_time = .4

    # In the factual world, a super-spreader event occured at time 0.5.
    # In the counterfactual world, however, this did not occur.
    factual = State(
        S=lambda s: s - torch.tensor(1.0),
        I=lambda i: i + torch.tensor(1.0),
    )
    superspreader_time = 0.1

    # We want to know how many people passengers would have bene infected at the
    #  time of landing had the super spreader event not occurred.

    with pyro.poutine.trace() as tr:
        with SimulatorEventLoop():
            with TwinWorldCounterfactual():
                with PointObservation(time=landing_time, data=landing_data):
                    with PointIntervention(time=superspreader_time, intervention=factual):
                        ret = simulate(
                            SimpleSIRDynamicsBayesReparam(),
                            init_state,
                            tspan,
                        )
                        assert False

    assert False
