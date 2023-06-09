import logging

import causal_pyro
import pyro
import pytest
import torch

from pyro.distributions import Normal, Uniform


import pyro
import torch
from pyro.distributions import constraints

from causal_pyro.dynamical.ops import State, simulate, Trajectory
from causal_pyro.dynamical.handlers import (
    ODEDynamics,
    PointInterruption,
    PointObservation,
    SimulatorEventLoop,
    simulate,
)

from .dynamical_fixtures import (
    sir_ode,
    check_trajectories_match,
    check_trajectories_match_in_all_but_values,
)

logger = logging.getLogger(__name__)

# Points at which to measure the state of the system.
tspan_values = torch.tensor([1.0, 2.0, 3.0])

# Initial state of the system.
init_state_values = State(
    S=torch.tensor(10.0), I=torch.tensor(3.0), R=torch.tensor(1.0)
)

obs_state = State(S=torch.tensor(5), I=torch.tensor(4), R=torch.tensor(5))

loglikelihood = lambda state: Normal(state.S, 1).log_prob(obs_state.S)

# Define intervention times before all tspan values.
condition_times = [2.0]

eps = 1e-3


@pytest.mark.parametrize("init_state", [init_state_values])
@pytest.mark.parametrize("tspan", [tspan_values])
@pytest.mark.parametrize("condition_time", condition_times)
def test_point_intervention_causes_difference(
    sir_ode, init_state, tspan, condition_time
):
    observational_execution_result = simulate(sir_ode, init_state, tspan)

    # Simulate with the intervention and ensure that the result differs from the observational execution.
    with SimulatorEventLoop():
        with PointObservation(time=condition_time, loglikelihood=loglikelihood):
            # if condition_time < tspan[0]:
            #     with pytest.raises(
            #         ValueError, match="is before the first time in the timespan"
            #     ):
            #         simulate(sir_ode, init_state, tspan)
            #     return
            # else:
            result_single_pint = simulate(sir_ode, init_state, tspan)

    assert check_trajectories_match_in_all_but_values(
        observational_execution_result, result_single_pint
    )


# Check that there is a log factor at the observed site
def test_log_factor_exists():
    init_state = State(S=torch.tensor(1.0), I=torch.tensor(2.0), R=torch.tensor(3.3))
    tspan = torch.tensor([1.0, 2.0, 3.0, 4.0])

    new_state = State(S=torch.tensor(10.0))
    S_obs = torch.tensor(10.0)
    loglikelihood = lambda state: Normal(state.S, 1).log_prob(S_obs)

    with pyro.poutine.trace() as tr:
        with SimulatorEventLoop():
            with PointObservation(time=2.9, loglikelihood=loglikelihood):
                simulate(sir_ode, init_state, tspan)

    time_key = tr.trace.nodes.keys().pop()
    assert isinstance(tr.trace.nodes[time_key]["fn"].log_factor, torch.Tensor)
