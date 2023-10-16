import logging

import pyro
import torch

from chirho.dynamical.handlers import InterruptionEventLoop, LogTrajectory
from chirho.dynamical.handlers.solver import TorchDiffEq
from chirho.dynamical.internals._utils import append
from chirho.dynamical.ops import State, simulate

from .dynamical_fixtures import bayes_sir_model, check_states_match

pyro.settings.set(module_local_params=True)

logger = logging.getLogger(__name__)

# Global variables for tests
init_state = State(S=torch.tensor(1.0), I=torch.tensor(2.0), R=torch.tensor(3.3))
start_time = torch.tensor(0.0)
end_time = torch.tensor(4.0)
logging_times = torch.tensor([1.0, 2.0, 3.0])


def test_logging():
    sir = bayes_sir_model()
    with LogTrajectory(
        times=logging_times,
    ) as dt1:
        result1 = simulate(sir, init_state, start_time, end_time, solver=TorchDiffEq())

    with LogTrajectory(
        times=logging_times,
    ) as dt2:
        with InterruptionEventLoop():
            result2 = simulate(
                sir, init_state, start_time, end_time, solver=TorchDiffEq()
            )
    result3 = simulate(sir, init_state, start_time, end_time, solver=TorchDiffEq())

    assert isinstance(result1, State)
    assert isinstance(dt1.trajectory, State)
    assert isinstance(dt2.trajectory, State)
    assert len(dt1.trajectory.keys) == 3
    assert len(dt2.trajectory.keys) == 3
    assert dt1.trajectory.keys == result1.keys
    assert dt2.trajectory.keys == result2.keys
    assert check_states_match(result1, result2)
    assert check_states_match(result1, result3)


def test_trajectory_methods():
    trajectory = State(S=torch.tensor([1.0, 2.0, 3.0]))
    assert trajectory.keys == frozenset({"S"})
    assert str(trajectory) == "State({'S': tensor([1., 2., 3.])})"


def test_append():
    trajectory1 = State(S=torch.tensor([1.0, 2.0, 3.0]))
    trajectory2 = State(S=torch.tensor([4.0, 5.0, 6.0]))
    trajectory = append(trajectory1, trajectory2)
    assert torch.allclose(
        trajectory.S, torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    ), "append() failed to append a trajectory"
