import logging

import pyro
import torch

from chirho.dynamical.handlers import TrajectoryLogging, SimulatorEventLoop
from chirho.dynamical.ops import State, simulate
from chirho.dynamical.ops.ODE.backends import TorchDiffEq

from .dynamical_fixtures import bayes_sir_model

pyro.settings.set(module_local_params=True)

logger = logging.getLogger(__name__)

# Global variables for tests
init_state = State(S=torch.tensor(1.0), I=torch.tensor(2.0), R=torch.tensor(3.3))
tspan = torch.tensor([0.0, 1.0, 2.0, 3.0, 4.0])


def test_logging():
    sir = bayes_sir_model()
    with TrajectoryLogging(tspan=tspan): # Not used yet. Will replace tspan in simulate() call.
        with SimulatorEventLoop():
            result = simulate(sir, init_state, tspan, backend=TorchDiffEq())
    
    assert result is not None
