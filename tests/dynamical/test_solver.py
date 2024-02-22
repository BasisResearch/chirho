import logging

import pyro
import pytest
import torch

from chirho.dynamical.handlers.solver import TorchDiffEq
from chirho.dynamical.ops import simulate

from .dynamical_fixtures import bayes_sir_model

pyro.settings.set(module_local_params=True)

logger = logging.getLogger(__name__)

# Global variables for tests
init_state = dict(S=torch.tensor(1.0), I=torch.tensor(2.0), R=torch.tensor(3.3))
start_time = torch.tensor(0.0)
end_time = torch.tensor(4.0)


def test_no_backend_error():
    sir = bayes_sir_model()
    with pytest.raises(NotImplementedError):
        simulate(sir, init_state, start_time, end_time)


def test_backend_arg():
    sir = bayes_sir_model()
    with TorchDiffEq():
        result = simulate(sir, init_state, start_time, end_time)
    assert result is not None


def test_torchdiffeq_broadcasting():
    with pyro.plate("plate", 3):
        sir = bayes_sir_model()
        with TorchDiffEq():
            result = simulate(sir, init_state, start_time, end_time)

    assert result is not None
