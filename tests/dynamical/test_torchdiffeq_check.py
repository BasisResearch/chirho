import logging

import pyro
import torch
import pytest

from chirho.dynamical.handlers.solver import TorchDiffEq
from chirho.dynamical.ops import State, simulate

from chirho.dynamical.internals.solver import check_dynamics
from chirho.dynamical.handlers.check_dynamics import RuntimeCheckDynamics

pyro.settings.set(module_local_params=True)

logger = logging.getLogger(__name__)

# Global variables for tests
init_state = State(S=torch.tensor(1.0))
start_time = torch.tensor(0.0)
end_time = torch.tensor(4.0)


class ValidModel:
    def diff(self, dstate: State, state: State) -> None:
        dstate.S = state.S


class InvalidModel:
    def diff(self, dstate: State, state: State) -> None:
        dstate.S = state.S + pyro.sample("dS_random", pyro.distributions.Normal(0, 1))


def test_torch_diffeq_check_dynamics():
    assert check_dynamics(TorchDiffEq(), ValidModel(), init_state, start_time)
    assert not check_dynamics(TorchDiffEq(), InvalidModel(), init_state, start_time)


def test_runtime_check_handler():
    
    with RuntimeCheckDynamics():
        result = simulate(ValidModel(), init_state, start_time, end_time, solver=TorchDiffEq())
    assert result is not None

    with pytest.raises(ValueError):
        with RuntimeCheckDynamics():
            simulate(InvalidModel(), init_state, start_time, end_time, solver=TorchDiffEq())