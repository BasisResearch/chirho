import logging

import pyro
import pytest
import torch

from chirho.dynamical.handlers import InterruptionEventLoop
from chirho.dynamical.handlers.solver import TorchDiffEq
from chirho.dynamical.ops_ import State, simulate

from .dynamical_fixtures import bayes_sir_model, check_states_match

pyro.settings.set(module_local_params=True)

logger = logging.getLogger(__name__)

# Global variables for tests
init_state = State(S=torch.tensor(1.0), I=torch.tensor(2.0), R=torch.tensor(3.3))
start_time = torch.tensor(0.0)
end_time = torch.tensor(4.0)


def test_no_backend_error():
    sir = bayes_sir_model()
    with pytest.raises(ValueError):
        simulate(sir, init_state, start_time, end_time)


def test_no_backend_SEL_error():
    sir = bayes_sir_model()
    with pytest.raises(ValueError):
        with InterruptionEventLoop():
            simulate(sir, init_state, start_time, end_time)


def test_backend_arg():
    sir = bayes_sir_model()
    with InterruptionEventLoop():
        result = simulate(sir, init_state, start_time, end_time, solver=TorchDiffEq())
    assert result is not None


def test_backend_handler():
    sir = bayes_sir_model()
    with InterruptionEventLoop():
        with TorchDiffEq():
            result_handler = simulate(sir, init_state, start_time, end_time)

        result_arg = simulate(
            sir, init_state, start_time, end_time, solver=TorchDiffEq()
        )

    assert check_states_match(result_handler, result_arg)
