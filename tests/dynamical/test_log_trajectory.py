import logging

import pyro
import torch

from chirho.dynamical.handlers import InterruptionEventLoop, LogTrajectory
from chirho.dynamical.handlers.solver import TorchDiffEq
from chirho.dynamical.internals._utils import append
from chirho.dynamical.ops import State, get_keys, simulate

from .dynamical_fixtures import bayes_sir_model, check_states_match

pyro.settings.set(module_local_params=True)

logger = logging.getLogger(__name__)

# Global variables for tests
init_state = State(
    time=torch.tensor(0.0),
    S=torch.tensor(1.0),
    I=torch.tensor(2.0),
    R=torch.tensor(3.3),
)
end_time = torch.tensor(4.0)
logging_times = torch.tensor([1.0, 2.0, 3.0])


def test_logging():
    sir = bayes_sir_model()
    with LogTrajectory(
        times=logging_times,
    ) as dt1:
        result1 = simulate(sir, init_state, end_time, solver=TorchDiffEq())

    with LogTrajectory(
        times=logging_times,
    ) as dt2:
        with InterruptionEventLoop():
            result2 = simulate(sir, init_state, end_time, solver=TorchDiffEq())
    result3 = simulate(sir, init_state, end_time, solver=TorchDiffEq())

    assert isinstance(result1, State)
    assert isinstance(dt1.trajectory, State)
    assert isinstance(dt2.trajectory, State)
    assert len(get_keys(dt1.trajectory)) == 4
    assert len(get_keys(dt2.trajectory)) == 4
    assert get_keys(dt1.trajectory) == get_keys(result1)
    assert get_keys(dt2.trajectory) == get_keys(result2)
    assert check_states_match(result1, result2)
    assert check_states_match(result1, result3)


def test_trajectory_methods():
    trajectory = State(
        time=torch.tensor([0.0, 1.0, 2.0]), S=torch.tensor([1.0, 2.0, 3.0])
    )
    assert get_keys(trajectory, include_time=False) == frozenset({"S"})
    assert get_keys(trajectory) == frozenset({"S", "time"})
    assert (
        str(trajectory)
        == "State({'time': tensor([0., 1., 2.]), 'S': tensor([1., 2., 3.])})"
    )


def test_append():
    trajectory1 = State(
        time=torch.tensor([0.0, 1.0, 2.0]), S=torch.tensor([1.0, 2.0, 3.0])
    )
    trajectory2 = State(
        time=torch.tensor([3.0, 4.0, 5.0]), S=torch.tensor([4.0, 5.0, 6.0])
    )
    trajectory = append(trajectory1, trajectory2)
    assert torch.allclose(
        trajectory.S, torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    ), "append() failed to append a trajectory"
    assert torch.allclose(
        trajectory.time, torch.tensor([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
    ), "append() failed to append a trajectory"
