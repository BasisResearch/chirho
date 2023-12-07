import logging

import pyro
import torch

from chirho.dynamical.handlers import LogTrajectory, StaticInterruption
from chirho.dynamical.handlers.solver import TorchDiffEq
from chirho.dynamical.internals._utils import append
from chirho.dynamical.ops import State, simulate

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


def test_logging():
    sir = bayes_sir_model()
    with TorchDiffEq():
        with LogTrajectory(times=logging_times) as dt1:
            result1 = simulate(sir, init_state, start_time, end_time)

    with LogTrajectory(times=logging_times) as dt2:
        with TorchDiffEq():
            result2 = simulate(sir, init_state, start_time, end_time)
    result3 = TorchDiffEq()(simulate)(sir, init_state, start_time, end_time)

    assert len(dt1.trajectory.keys()) == 3
    assert len(dt2.trajectory.keys()) == 3
    assert dt1.trajectory.keys() == result1.keys()
    assert dt2.trajectory.keys() == result2.keys()
    assert check_states_match(dt1.trajectory, dt2.trajectory)
    assert check_states_match(result1, result2)
    assert check_states_match(result1, result3)


def test_logging_with_colliding_interruption():
    sir = bayes_sir_model()
    with LogTrajectory(times=logging_times) as dt1:
        with TorchDiffEq():
            simulate(sir, init_state, start_time, end_time)

    with LogTrajectory(times=logging_times) as dt2:
        with TorchDiffEq():
            with StaticInterruption(
                time=torch.tensor(2.0),
            ):
                simulate(sir, init_state, start_time, end_time)

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


def test_start_end_time_collisions():
    def dynamics(s: State) -> State:
        return dict(X=s["X"] * (1 - s["X"]))

    init_state = dict(X=torch.tensor(0.5))
    start_time, end_time = torch.tensor(0.0), torch.tensor(3.0)

    with TorchDiffEq():
        with LogTrajectory(times=torch.tensor([0.0, 1.0, 2.0, 3.0])) as log1:
            simulate(dynamics, init_state, start_time, end_time)

    with LogTrajectory(times=torch.tensor([0.0, 1.0, 2.0, 3.0])) as log2:
        with TorchDiffEq():
            simulate(dynamics, init_state, start_time, end_time)

    assert check_states_match(log1.trajectory, log2.trajectory)

    assert (
        len(log1.trajectory["X"])
        == len(log1.times)
        == len(log2.trajectory["X"])
        == len(log2.times)
        == 4
    )  # previously failed bc len(X) == 3


def test_multiple_simulates():
    sir1 = bayes_sir_model()
    sir2 = bayes_sir_model()

    assert sir1.beta != sir2.beta
    assert sir1.gamma != sir2.gamma

    with LogTrajectory(times=logging_times) as dt1:
        with TorchDiffEq():
            result11 = simulate(sir1, init_state, start_time, end_time)
            result12 = simulate(sir2, init_state, start_time, end_time)

    with LogTrajectory(times=logging_times) as dt2:
        with TorchDiffEq():
            result21 = simulate(sir1, init_state, start_time, end_time)

    with LogTrajectory(times=logging_times) as dt3:
        with TorchDiffEq():
            result22 = simulate(sir2, init_state, start_time, end_time)

    # Simulation outputs do not depend on LogTrajectory context
    assert check_states_match(result11, result21)
    assert check_states_match(result12, result22)

    # LogTrajectory trajectory only preserves the final `simulate` call.
    assert check_trajectories_match_in_all_but_values(dt1.trajectory, dt2.trajectory)
    assert check_states_match(dt1.trajectory, dt3.trajectory)
