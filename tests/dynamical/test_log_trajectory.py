import logging

import pyro
import torch

from chirho.dynamical.handlers import LogTrajectory, StaticInterruption
from chirho.dynamical.handlers.solver import TorchDiffEq
from chirho.dynamical.internals._utils import append
from chirho.dynamical.ops import State, simulate
import pytest

from .dynamical_fixtures import (
    bayes_sir_model,
    check_states_match,
    check_trajectories_match_in_all_but_values,
)

pyro.settings.set(module_local_params=True)

logger = logging.getLogger(__name__)

# Global variables for tests
init_state = dict(S=torch.tensor(1.0), I=torch.tensor(2.0), R=torch.tensor(3.3))
start_time = torch.tensor(0.0)
end_time = torch.tensor(4.0)
logging_times = torch.tensor([1.0, 2.0, 3.0])


@pytest.mark.parametrize("solver,dynamics,simulate_kwargs", [
    (TorchDiffEq, bayes_sir_model(), dict())
])
def test_logging(solver, dynamics, simulate_kwargs):
    with solver():
        with LogTrajectory(times=logging_times) as dt1:
            result1 = simulate(dynamics, init_state, start_time, end_time, **simulate_kwargs)

    with LogTrajectory(times=logging_times) as dt2:
        with solver():
            result2 = simulate(dynamics, init_state, start_time, end_time, **simulate_kwargs)
    result3 = solver()(simulate)(dynamics, init_state, start_time, end_time, **simulate_kwargs)

    assert len(dt1.trajectory.keys()) == 3
    assert len(dt2.trajectory.keys()) == 3
    assert dt1.trajectory.keys() == result1.keys()
    assert dt2.trajectory.keys() == result2.keys()
    assert check_states_match(dt1.trajectory, dt2.trajectory)
    assert check_states_match(result1, result2)
    assert check_states_match(result1, result3)


@pytest.mark.parametrize("solver,dynamics,simulate_kwargs", [
    (TorchDiffEq, bayes_sir_model(), dict())
])
def test_logging_with_colliding_interruption(solver, dynamics, simulate_kwargs):
    with LogTrajectory(times=logging_times) as dt1:
        with solver():
            simulate(dynamics, init_state, start_time, end_time, **simulate_kwargs)

    with LogTrajectory(times=logging_times) as dt2:
        with solver():
            with StaticInterruption(
                time=torch.tensor(2.0),
            ):
                simulate(dynamics, init_state, start_time, end_time, **simulate_kwargs)

    check_states_match(dt1.trajectory, dt2.trajectory)


def test_trajectory_methods():
    trajectory = dict(S=torch.tensor([1.0, 2.0, 3.0]))
    assert trajectory.keys() == frozenset({"S"})


def test_append():
    trajectory1 = dict(S=torch.tensor([1.0, 2.0, 3.0]))
    trajectory2 = dict(S=torch.tensor([4.0, 5.0, 6.0]))
    trajectory = append(trajectory1, trajectory2)
    assert torch.allclose(
        trajectory["S"], torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    ), "append() failed to append a trajectory"


@pytest.mark.parametrize("solver,dynamics,simulate_kwargs", [
    (TorchDiffEq, lambda s: dict(X=s["X"] * (1 - s["X"])), dict())
])
def test_start_end_time_collisions(solver, dynamics, simulate_kwargs):
    start_time, end_time = torch.tensor(0.0), torch.tensor(3.0)
    init_state = dict(X=torch.tensor(0.5))

    with solver():
        with LogTrajectory(times=torch.tensor([0.0, 1.0, 2.0, 3.0])) as log1:
            simulate(dynamics, init_state, start_time, end_time)

    with LogTrajectory(times=torch.tensor([0.0, 1.0, 2.0, 3.0])) as log2:
        with solver():
            simulate(dynamics, init_state, start_time, end_time)

    assert check_states_match(log1.trajectory, log2.trajectory)

    assert (
        len(log1.trajectory["X"])
        == len(log1.times)
        == len(log2.trajectory["X"])
        == len(log2.times)
        == 4
    )  # previously failed bc len(X) == 3


@pytest.mark.parametrize("solver,dynamics1,simulate_kwargs1,dynamics2,simulate_kwargs2", [
    (TorchDiffEq, bayes_sir_model(), dict(), bayes_sir_model(), dict())
])
def test_multiple_simulates(solver, dynamics1, simulate_kwargs1, dynamics2, simulate_kwargs2):

    with LogTrajectory(times=logging_times) as dt1:
        with solver():
            result11 = simulate(dynamics1, init_state, start_time, end_time)
            result12 = simulate(dynamics2, init_state, start_time, end_time)

    with LogTrajectory(times=logging_times) as dt2:
        with solver():
            result21 = simulate(dynamics1, init_state, start_time, end_time)

    with LogTrajectory(times=logging_times) as dt3:
        with solver():
            result22 = simulate(dynamics2, init_state, start_time, end_time)

    # Simulation outputs do not depend on LogTrajectory context
    assert check_states_match(result11, result21)
    assert check_states_match(result12, result22)

    # These are run with different dynamics, so make sure the two trajectories are different.
    with pytest.raises(AssertionError):
        check_states_match(result11, result12)
    with pytest.raises(AssertionError):
        check_states_match(result21, result22)

    # LogTrajectory trajectory only preserves the final `simulate` call.
    assert check_trajectories_match_in_all_but_values(dt1.trajectory, dt2.trajectory)
    assert check_states_match(dt1.trajectory, dt3.trajectory)
