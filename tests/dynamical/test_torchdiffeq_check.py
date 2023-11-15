import logging

import pyro
import pytest
import torch

from chirho.dynamical.handlers.solver import TorchDiffEq
from chirho.dynamical.internals.solver import validate_dynamics
from chirho.dynamical.ops import State

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


def test_validate_dynamics_torchdiffeq():
    with TorchDiffEq():
        validate_dynamics(
            valid_diff,
            init_state,
            start_time,
            end_time,
        )

    with pytest.raises(ValueError):
        with TorchDiffEq():
            validate_dynamics(
                invalid_diff,
                init_state,
                start_time,
                end_time,
            )
