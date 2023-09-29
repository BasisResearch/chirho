import logging

import pyro
import torch

from chirho.dynamical.handlers import DynamicTrace, SimulatorEventLoop
from chirho.dynamical.handlers.ODE.solvers import TorchDiffEq
from chirho.dynamical.ops import State, Trajectory, simulate

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
    ) as dt:
        result1 = simulate(sir, init_state, start_time, end_time, solver=TorchDiffEq())

        with SimulatorEventLoop():
            result2 = simulate(
                sir, init_state, start_time, end_time, solver=TorchDiffEq()
            )
    result3 = simulate(sir, init_state, start_time, end_time, solver=TorchDiffEq())
    
    assert isinstance(result1, State)
    assert isinstance(dt.trace, Trajectory)
    assert len(dt.trace.keys) == 3
    assert (dt.trace.keys == result1.keys)
    assert check_states_match(result1, result2)
    assert check_states_match(result1, result3)
