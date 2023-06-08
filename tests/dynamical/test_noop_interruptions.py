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
    PointIntervention,
    SimulatorEventLoop,
    simulate,
)

from .dynamical_fixtures import sir_ode, check_trajectories_match

logger = logging.getLogger(__name__)

# Points at which to measure the state of the system.
tspan_values = torch.tensor([1.0, 2.0, 3.0, 4.0])
# Initial state of the system.
init_state_values = State(S=torch.tensor(10.0), I=torch.tensor(3.0), R=torch.tensor(1.0))

eps = 1e-3

intervene_states = [
    State(S=torch.tensor(11.0)),
    State(I=torch.tensor(9.0)),
    State(S=torch.tensor(10.0), R=torch.tensor(5.0)),
    State(S=torch.tensor(20.0), I=torch.tensor(11.0), R=torch.tensor(4.0)),
]


@pytest.mark.parametrize("init_state", [init_state_values])
@pytest.mark.parametrize("tspan", [tspan_values])
def test_noop_point_interruptions(sir_ode, init_state, tspan):

    observational_execution_result = simulate(sir_ode, init_state, tspan)

    # Test with standard point interruptions within timespan.
    with SimulatorEventLoop():
        with PointInterruption(time=tspan[-1] / 2. + eps):
            result_pint = simulate(sir_ode, init_state, tspan)

    assert check_trajectories_match(observational_execution_result, result_pint)

    # Test with two standard point interruptions.
    with SimulatorEventLoop():
        with PointInterruption(time=tspan[-1] / 4. + eps):  # roughly 1/4 of the way through the timespan
            with PointInterruption(time=(tspan[-1] / 4.) * 3 + eps):  # roughly 3/4
                result_double_pint1 = simulate(sir_ode, init_state, tspan)

    # FIXME AZ-yu28184 This test fails rn because the state of the system at the the point interruption is included in the
    #  returned vector of measurements. TODO parse that out so that user gets what they ask for?
    #  Odd that this only procs for the double point interruption case
    assert check_trajectories_match(observational_execution_result, result_double_pint1)

    # Test with two standard point interruptions, in a different order.
    with SimulatorEventLoop():
        with PointInterruption(time=(tspan[-1] / 4.) * 3 + eps):  # roughly 3/4
            with PointInterruption(time=tspan[-1] / 4. + eps):  # roughly 1/3
                result_double_pint2 = simulate(sir_ode, init_state, tspan)

    assert check_trajectories_match(observational_execution_result, result_double_pint2)

    # TODO test pointinterruptions when they are out of scope of the timespan


@pytest.mark.parametrize("init_state", [init_state_values])
@pytest.mark.parametrize("tspan", [tspan_values])
@pytest.mark.parametrize("intervene_state", intervene_states)
def test_noop_point_interventions(sir_ode, init_state, tspan, intervene_state):
    """
    Test whether point interruptions that don't intervene match the unhandled ("observatonal") default simulation.
    :return:
    """

    post_measurement_intervention_time = tspan_values.max() + 1.0

    observational_execution_result = simulate(sir_ode, init_state, tspan)

    # Test a single point intervention.
    with SimulatorEventLoop():
        with PointIntervention(time=post_measurement_intervention_time, intervention=intervene_state):
            result_single_pi = simulate(sir_ode, init_state, tspan)

    assert check_trajectories_match(observational_execution_result, result_single_pi)

    # Test two point interventions out of scope.
    with SimulatorEventLoop():
        with PointIntervention(time=post_measurement_intervention_time, intervention=intervene_state):
            with PointIntervention(time=post_measurement_intervention_time + 1.0, intervention=intervene_state):
                result_double_pi1 = simulate(sir_ode, init_state, tspan)

    assert check_trajectories_match(observational_execution_result, result_double_pi1)

    # Test with two point interventions out of scope, in a different order.
    with SimulatorEventLoop():
        with PointIntervention(time=post_measurement_intervention_time + 1.0, intervention=intervene_state):
            with PointIntervention(time=post_measurement_intervention_time, intervention=intervene_state):
                result_double_pi2 = simulate(sir_ode, init_state, tspan)

    assert check_trajectories_match(observational_execution_result, result_double_pi2)
