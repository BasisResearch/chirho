import logging

import pyro
import pytest
import torch

from chirho.dynamical.handlers.solver import RuntimeCheckTorchDiffEq
from chirho.dynamical.ops import State, simulate

pyro.settings.set(module_local_params=True)

logger = logging.getLogger(__name__)

# Global variables for tests
init_state = State(S=torch.tensor(1.0))
start_time = torch.tensor(0.0)
end_time = torch.tensor(4.0)


def valid_diff(state: State) -> State:
    return state


def invalid_diff(state: State) -> State:
    x = pyro.sample("x", pyro.distributions.Normal(0.0, 1.0))
    return State(S=(state["S"] + x))


def test_runtime_check_handler():
    result = simulate(
        valid_diff,
        init_state,
        start_time,
        end_time,
        solver=RuntimeCheckTorchDiffEq(),
    )
    assert result is not None

    with pytest.raises(ValueError):
        simulate(
            invalid_diff,
            init_state,
            start_time,
            end_time,
            solver=RuntimeCheckTorchDiffEq(),
        )
