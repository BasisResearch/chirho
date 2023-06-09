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
    SimulatorEventLoop,
    PointIntervention,
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

# Large interventions that will make a difference.
intervene_states = [
    State(I=torch.tensor(50.0)),
    State(S=torch.tensor(50.0), R=torch.tensor(50.0)),
    State(S=torch.tensor(50.0), I=torch.tensor(50.0), R=torch.tensor(50.0)),
]

# Define intervention times before all tspan values.
intervene_times = tspan_values - 0.5


eps = 1e-3


@pytest.mark.parametrize("init_state", [init_state_values])
@pytest.mark.parametrize("tspan", [tspan_values])
@pytest.mark.parametrize("intervene_state", intervene_states)
@pytest.mark.parametrize("intervene_time", intervene_times)
def test_point_intervention_causes_difference(
    sir_ode, init_state, tspan, intervene_state, intervene_time
):
    observational_execution_result = simulate(sir_ode, init_state, tspan)

    # Simulate with the intervention and ensure that the result differs from the observational execution.
    with SimulatorEventLoop():
        with PointIntervention(time=intervene_time, intervention=intervene_state):
            if intervene_time < tspan[0]:
                with pytest.raises(
                    ValueError, match="is before the first time in the timespan"
                ):
                    simulate(sir_ode, init_state, tspan)
                return
            else:
                result_single_pint = simulate(sir_ode, init_state, tspan)

    assert check_trajectories_match_in_all_but_values(
        observational_execution_result, result_single_pint
    )


# TODO test what happens when the intervention time is exactly at the start of the time span.


# TODO get rid of some entries cz this test takes too long to run w/ all permutations.
@pytest.mark.parametrize("init_state", [init_state_values])
@pytest.mark.parametrize("tspan", [tspan_values])
@pytest.mark.parametrize("intervene_state1", intervene_states)
@pytest.mark.parametrize("intervene_time1", intervene_times)
@pytest.mark.parametrize("intervene_state2", intervene_states)
@pytest.mark.parametrize("intervene_time2", intervene_times)
def test_nested_point_interventions_cause_difference(
    sir_ode,
    init_state,
    tspan,
    intervene_state1,
    intervene_time1,
    intervene_state2,
    intervene_time2,
):
    observational_execution_result = simulate(sir_ode, init_state, tspan)

    # Simulate with the intervention and ensure that the result differs from the observational execution.
    with SimulatorEventLoop():
        with PointIntervention(time=intervene_time1, intervention=intervene_state1):
            with PointIntervention(time=intervene_time2, intervention=intervene_state2):
                if intervene_time1 < tspan[0] or intervene_time2 < tspan[0]:
                    with pytest.raises(
                        ValueError, match="is before the first time in the timespan"
                    ):
                        simulate(sir_ode, init_state, tspan)
                    return
                elif torch.isclose(intervene_time1, intervene_time2):
                    with pytest.raises(
                        ValueError,
                        match="Two point interruptions cannot occur at the same time.",
                    ):
                        simulate(sir_ode, init_state, tspan)
                    return
                else:
                    result_nested_pint = simulate(sir_ode, init_state, tspan)

    assert check_trajectories_match_in_all_but_values(
        observational_execution_result, result_nested_pint
    )

    # Don't need to flip order b/c the argument permutation will effectively do this for us.


# TODO test that we're getting the exactly right answer, instead of just "a different answer" as we are now.
