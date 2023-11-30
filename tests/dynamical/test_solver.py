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
    with TorchDiffEq():  # TODO parameterize and use diffeqdotjl
        result = simulate(sir, init_state, start_time, end_time)
    assert result is not None


@pytest.mark.parametrize("simulate_inner_bknd", [_diffeqdotjl_ode_simulate_inner, _torchdiffeq_ode_simulate_inner])
def test_inner_simulates_of_solvers_match_forward(simulate_inner_bknd):
    sp0 = State(x=torch.tensor(10.).double(), c=torch.tensor(0.1).double())
    timespan = torch.linspace(0., 10., 10).double()

    res = simulate_inner_bknd(lambda s: State(x=-s['x'] * s['c']), sp0, timespan)

    correct = torch.tensor([10.0000,  8.9484,  8.0074,  7.1653,  6.4118,  5.7375,  5.1342,  4.5943,
                            4.1111,  3.6788], dtype=torch.float64)

    assert torch.allclose(res['x'], correct, atol=1e-4)


@pytest.mark.parametrize(
    "simulate_inner_bknd", [_diffeqdotjl_ode_simulate_inner, _torchdiffeq_ode_simulate_inner])
@pytest.mark.parametrize("x0", [torch.tensor(10.), torch.tensor([10., 5.])])
@pytest.mark.parametrize("c", [torch.tensor(0.1), torch.tensor([0.1, 0.2])])
def test_smoke_inner_simulates_forward_nd(simulate_inner_bknd, x0, c):
    sp0 = State(x=x0.double(), c=c.double())
    timespan = torch.linspace(0., 10., 10).double()

    simulate_inner_bknd(lambda s: State(x=-s['x'] * s['c']), sp0, timespan)


@pytest.mark.parametrize(
    "simulate_inner_bknd", [_diffeqdotjl_ode_simulate_inner, _torchdiffeq_ode_simulate_inner])
@pytest.mark.parametrize("x0", [torch.tensor(10.), torch.tensor([10., 5.])])
@pytest.mark.parametrize("c_", [torch.tensor(0.1), torch.tensor([0.1, 0.2])])
def test_gradcheck_inner_simulates_of_solvers_wrt_param(simulate_inner_bknd, x0, c_):
    c_ = c_.double().requires_grad_()
    timespan = torch.linspace(0., 10., 10).double()

    # TODO add other inputs we want to gradcheck.

    def dynamics(s: State):
        return State(x=-(s['x'] * s['c']))

    def wrapped_simulate_inner(c):
        sp0 = State(x=x0.double(), c=c)
        return simulate_inner_bknd(dynamics, sp0, timespan)['x']

    torch.autograd.gradcheck(wrapped_simulate_inner, c_)


# TODO test whether float64 is properly required of initial state for diffeqpy backend.
# TODO test whether float64 is properly required of returned dstate even if initial state passes float64 check.
