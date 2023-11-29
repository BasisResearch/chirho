import juliacall  # Must precede even indirect torch imports to prevent segfault.

import logging

import numpy as np
import pyro
import pytest
import torch

from chirho.dynamical.handlers.solver import TorchDiffEq
from chirho.dynamical.ops import State, simulate
from chirho.dynamical.internals.backends.diffeqdotjl import _diffeqdotjl_ode_simulate_inner
from chirho.dynamical.internals.backends.torchdiffeq import _torchdiffeq_ode_simulate_inner

from .dynamical_fixtures import bayes_sir_model

pyro.settings.set(module_local_params=True)

logger = logging.getLogger(__name__)

# Global variables for tests
init_state = State(S=torch.tensor(1.0), I=torch.tensor(2.0), R=torch.tensor(3.3))
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


def test_inner_simulates_of_solvers_match():
    sp0 = State(x=torch.tensor(10.), c=torch.tensor(0.1))
    timespan = torch.linspace(0., 10., 10)

    diffeqdotjl_res = _diffeqdotjl_ode_simulate_inner(
        dynamics=lambda s: State(x=-s['x'] * s['c']),
        initial_state_and_params=sp0,
        timespan=timespan
    )

    torchdiffeq_res = _torchdiffeq_ode_simulate_inner(
        dynamics=lambda s: State(x=-s['x'] * s['c']),
        initial_state=sp0,
        timespan=timespan
    )

    assert torch.allclose(diffeqdotjl_res['x'], torchdiffeq_res['x'])

    # TODO gradcheck.
