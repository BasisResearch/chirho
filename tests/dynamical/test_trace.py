import logging

import pyro
import torch

from chirho.dynamical.handlers import DynamicTrace, SimulatorEventLoop
from chirho.dynamical.handlers.solver import TorchDiffEq
from chirho.dynamical.ops import State, simulate
from chirho.dynamical.ops.trajectory import Trajectory

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
    with DynamicTrace(
        logging_times=logging_times,
    ) as dt1:
        result1 = simulate(sir, init_state, start_time, end_time, solver=TorchDiffEq())

    with DynamicTrace(
        logging_times=logging_times,
    ) as dt2:
        with SimulatorEventLoop():
            result2 = simulate(
                sir, init_state, start_time, end_time, solver=TorchDiffEq()
            )
    result3 = simulate(sir, init_state, start_time, end_time, solver=TorchDiffEq())

    assert isinstance(result1, State)
    assert isinstance(dt1.trace, Trajectory)
    assert isinstance(dt2.trace, Trajectory)
    assert len(dt1.trace.keys) == 3
    assert len(dt2.trace.keys) == 3
    assert dt1.trace.keys == result1.keys
    assert dt2.trace.keys == result2.keys
    assert check_states_match(result1, result2)
    assert check_states_match(result1, result3)
