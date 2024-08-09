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


@pytest.mark.parametrize("backend", [TorchDiffEq])
@pytest.mark.parametrize("dynamics", [bayes_sir_model()])
def test_backend_arg(backend, dynamics):
    with backend():
        result = simulate(dynamics, init_state, start_time, end_time)
    assert result is not None


@pytest.mark.parametrize("backend", [TorchDiffEq])
@pytest.mark.parametrize("dynamics_builder", [bayes_sir_model])
def test_broadcasting(backend, dynamics_builder):
    with pyro.plate("plate", 3):
        dynamics = dynamics_builder()
        with backend():
            result = simulate(dynamics, init_state, start_time, end_time)

    for v in result.values():
        assert v.shape == (3,)

    assert result is not None
