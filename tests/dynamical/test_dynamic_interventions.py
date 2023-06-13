import logging

import causal_pyro
import pyro
import pytest
import torch

from pyro.distributions import Normal, Uniform
import numpy as np


import pyro
import torch
from pyro.distributions import constraints

from causal_pyro.dynamical.ops import State, simulate, Trajectory
from causal_pyro.dynamical.handlers import (
    ODEDynamics,
    PointInterruption,
    SimulatorEventLoop,
    PointIntervention,
    DynamicIntervention,
    simulate,
)

from .dynamical_fixtures import (
    sir_ode,
    check_trajectories_match,
    check_trajectories_match_in_all_but_values,
)

logger = logging.getLogger(__name__)

# Points at which to measure the state of the system.
tspan_values = torch.arange(0.0, 3.0, 0.03)

# Initial state of the system.
init_state = State(S=torch.tensor(50.0), I=torch.tensor(3.0), R=torch.tensor(0.0))

# State at which the dynamic intervention will trigger.
trigger_state = State(R=torch.tensor(30.0))

# State we'll switch to when the dynamic intervention triggers.
intervene_state = State(S=torch.tensor(50.0))


def get_state_reached_event_f(target_state: State[torch.tensor]):

    def event_f(t: torch.tensor, state: State[torch.tensor]):
        # ret = target_state.subtract_shared_variables(state).l2()

        return state.R - target_state.R

    return event_f


@pytest.mark.parametrize("init_state", [init_state])
@pytest.mark.parametrize("tspan", [tspan_values])
@pytest.mark.parametrize("trigger_state", [trigger_state])
@pytest.mark.parametrize("intervene_state", [intervene_state])
def test_dynamic_intervention_runtime(
        sir_ode, init_state, tspan, trigger_state, intervene_state):

    with SimulatorEventLoop():
        with DynamicIntervention(
                event_f=get_state_reached_event_f(trigger_state),
                intervention=intervene_state,
                var_order=init_state.var_order,
                num_applications=1):
            res = simulate(sir_ode, init_state, tspan)

    assert True
