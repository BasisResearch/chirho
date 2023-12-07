import logging

import pyro
import torch
import pytest

from chirho.dynamical.handlers import LogTrajectory, StaticInterruption
from chirho.dynamical.handlers.solver import TorchDiffEq, DiffEqDotJL
from chirho.dynamical.internals._utils import append
from chirho.dynamical.ops import State, simulate

from .dynamical_fixtures import build_bayes_sir_dynamics, check_states_match

pyro.settings.set(module_local_params=True)

logger = logging.getLogger(__name__)

# TODO do17bdy1t .double is strictly for DiffEqDotJL backend, remove when float32 is supported.
# Global variables for tests
init_state = State(S=torch.tensor(1.0).double(), I=torch.tensor(2.0).double(), R=torch.tensor(3.3).double())
start_time = torch.tensor(0.0).double()
end_time = torch.tensor(4.0).double()
logging_times = torch.tensor([1.0, 2.0, 3.0]).double()


@pytest.mark.parametrize("solver", [TorchDiffEq, DiffEqDotJL])
@pytest.mark.parametrize("build_dynamics", [build_bayes_sir_dynamics])
def test_logging(solver, build_dynamics):
    dynamics = build_dynamics(solver)
    dynamics.extend_initial_state_with_params_(init_state)

    with solver(), LogTrajectory(times=logging_times) as dt1:
        result1 = simulate(dynamics, init_state, start_time, end_time)

    with LogTrajectory(times=logging_times) as dt2:
        with solver():
            result2 = simulate(dynamics, init_state, start_time, end_time)
    result3 = solver()(simulate)(dynamics, init_state, start_time, end_time)

    assert len(dt1.trajectory.keys()) == 3
    assert len(dt2.trajectory.keys()) == 3
    assert dt1.trajectory.keys() == result1.keys()
    assert dt2.trajectory.keys() == result2.keys()
    assert check_states_match(result1, result2)
    assert check_states_match(result1, result3)


@pytest.mark.parametrize("solver", [TorchDiffEq, DiffEqDotJL])
@pytest.mark.parametrize("build_dynamics", [build_bayes_sir_dynamics])
def test_logging_with_colliding_interruption(solver, build_dynamics):
    dynamics = build_dynamics(solver)
    dynamics.extend_initial_state_with_params_(init_state)

    with solver(), LogTrajectory(times=logging_times) as dt1:
        simulate(dynamics, init_state, start_time, end_time)

    with LogTrajectory(times=logging_times) as dt2:
        with solver():
            with StaticInterruption(
                time=torch.tensor(2.0),
            ):
                simulate(dynamics, init_state, start_time, end_time)

    check_states_match(dt1.trajectory, dt2.trajectory)


def test_trajectory_methods():
    trajectory = State(S=torch.tensor([1.0, 2.0, 3.0]))
    assert trajectory.keys() == frozenset({"S"})


def test_append():
    trajectory1 = State(S=torch.tensor([1.0, 2.0, 3.0]))
    trajectory2 = State(S=torch.tensor([4.0, 5.0, 6.0]))
    trajectory = append(trajectory1, trajectory2)
    assert torch.allclose(
        trajectory["S"], torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    ), "append() failed to append a trajectory"
