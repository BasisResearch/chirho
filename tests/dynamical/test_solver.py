import logging

import pyro
import pytest
import torch

from chirho.dynamical.handlers import SimulatorEventLoop
from chirho.dynamical.handlers.ODE.solvers import TorchDiffEq
from chirho.dynamical.ops import State, simulate

from .dynamical_fixtures import bayes_sir_model, check_trajectories_match

pyro.settings.set(module_local_params=True)

logger = logging.getLogger(__name__)

# Global variables for tests
init_state = State(S=torch.tensor(1.0), I=torch.tensor(2.0), R=torch.tensor(3.3))
tspan = torch.tensor([0.0, 1.0, 2.0, 3.0, 4.0])


def test_no_backend_error():
    sir = bayes_sir_model()
    with pytest.raises(ValueError):
        simulate(sir, init_state, tspan)


def test_no_backend_SEL_error():
    sir = bayes_sir_model()
    with pytest.raises(ValueError):
        with SimulatorEventLoop():
            simulate(sir, init_state, tspan)


def test_backend_arg():
    sir = bayes_sir_model()
    with SimulatorEventLoop():
        result = simulate(sir, init_state, tspan, solver=TorchDiffEq())
    assert result is not None


def test_backend_handler():
    sir = bayes_sir_model()
    with SimulatorEventLoop():
        with TorchDiffEq():
            result_handler = simulate(sir, init_state, tspan)

        result_arg = simulate(sir, init_state, tspan, solver=TorchDiffEq())

    assert check_trajectories_match(result_handler, result_arg)
