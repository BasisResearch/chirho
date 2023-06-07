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
    simulate,
)

logger = logging.getLogger(__name__)


class SimpleSIRDynamics(ODEDynamics):
    @pyro.nn.PyroParam(constraint=constraints.positive)
    def beta(self):
        return torch.tensor(0.5)

    @pyro.nn.PyroParam(constraint=constraints.positive)
    def gamma(self):
        return torch.tensor(0.7)

    def diff(self, dX: State[torch.Tensor], X: State[torch.Tensor]):
        dX.S = -self.beta * X.S * X.I
        dX.I = self.beta * X.S * X.I - self.gamma * X.I
        dX.R = self.gamma * X.I


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


def check_trajectories_match(traj1: Trajectory[torch.tensor], traj2: Trajectory[torch.tensor]):

    assert traj2.keys == traj1.keys, "Trajectories have different state variables."

    for k in traj1.keys:
        assert len(getattr(traj2, k)) == len(getattr(traj1, k)),\
            f"Trajectories have different lengths for variable {k}."
        assert torch.allclose(getattr(traj2, k), getattr(traj1, k)),\
            f"Trajectories differ in state trajectory of variable {k}."

    return True


@pytest.mark.parametrize("init_state", [init_state_values])
@pytest.mark.parametrize("tspan", [tspan_values])
def test_noop_point_interruptions(init_state, tspan):
    SIR_simple_model = SimpleSIRDynamics()

    observational_execution_result = simulate(SIR_simple_model, init_state, tspan)

    # Test with standard point interruptions within timespan.
    with PointInterruption(time=tspan[-1] / 2. + eps):
        result_pint = simulate(SIR_simple_model, init_state, tspan)

    assert check_trajectories_match(observational_execution_result, result_pint)

    # Test with two standard point interruptions.
    with PointInterruption(time=tspan[-1] / 4. + eps):  # roughly 1/4 of the way through the timespan
        with PointInterruption(time=(tspan[-1] / 4.) * 3 + eps):  # roughly 3/4
            result_double_pint = simulate(SIR_simple_model, init_state, tspan)

    # FIXME AZ-yu28184 This test fails rn because the state of the system at the the point interruption is included in the
    #  returned vector of measurements. TODO parse that out so that user gets what they ask for?
    #  Odd that this only procs for the double point interruption case
    assert check_trajectories_match(observational_execution_result, result_double_pint)

    # Test with two standard point interruptions, in a different order.
    with PointInterruption(time=(tspan[-1] / 4.) * 3 + eps):  # roughly 3/4
        with PointInterruption(time=tspan[-1] / 4. + eps):  # roughly 1/3
            result_double_pint = simulate(SIR_simple_model, init_state, tspan)

    assert check_trajectories_match(observational_execution_result, result_double_pint)


@pytest.mark.parametrize("init_state", [init_state_values])
@pytest.mark.parametrize("tspan", [tspan_values])
@pytest.mark.parametrize("intervene_state", intervene_states)
def test_noop_point_interventions(init_state, tspan, intervene_state):
    """
    Test whether point interruptions that don't intervene match the unhandled ("observatonal") default simulation.
    :return:
    """

    SIR_simple_model = SimpleSIRDynamics()

    post_measurement_intervention_time = tspan_values.max() + 1.0

    observational_execution_result = simulate(SIR_simple_model, init_state, tspan)

    # Test a single point intervention.
    with PointIntervention(time=post_measurement_intervention_time, intervention=intervene_state):
        result_single_pi = simulate(SIR_simple_model, init_state, tspan)

    assert check_trajectories_match(observational_execution_result, result_single_pi)

    # Test two point interventions out of scope.
    with PointIntervention(time=post_measurement_intervention_time, intervention=intervene_state):
        with PointIntervention(time=post_measurement_intervention_time + 1.0, intervention=intervene_state):
            result_double_pi1 = simulate(SIR_simple_model, init_state, tspan)

    assert check_trajectories_match(observational_execution_result, result_double_pi1)

    # Test with two point interventions out of scope, in a different order.
    with PointIntervention(time=post_measurement_intervention_time + 1.0, intervention=intervene_state):
        with PointIntervention(time=post_measurement_intervention_time, intervention=intervene_state):
            result_double_pi2 = simulate(SIR_simple_model, init_state, tspan)

    assert check_trajectories_match(observational_execution_result, result_double_pi2)
