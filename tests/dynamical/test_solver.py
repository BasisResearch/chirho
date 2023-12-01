import juliacall  # Must precede even indirect torch imports to prevent segfault.

import logging

import numpy as np
import pyro
import pytest
import torch

from chirho.dynamical.handlers.solver import TorchDiffEq, DiffEqDotJL
from chirho.dynamical.ops import State, simulate
from chirho.dynamical.internals.backends.diffeqdotjl import _diffeqdotjl_ode_simulate_inner
from chirho.dynamical.internals.backends.torchdiffeq import _torchdiffeq_ode_simulate_inner
from chirho.dynamical.handlers import LogTrajectory

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


@pytest.mark.parametrize("solver", [DiffEqDotJL, TorchDiffEq])
def test_inner_simulates_of_solvers_match_forward(solver):
    sp0 = State(x=torch.tensor(10.).double(), c=torch.tensor(0.1).double())
    timespan = torch.linspace(1.0, 10., 10).double()

    def dynamics(s: State):
        # FIXME
        #  -(s['x'] * s['c']) works but (-s['x']) * s['c'] errors on exceeding max ndim of 32. This occurs
        #  despite the fact that at every stage everything is a proper ndarray of dtype object.
        return State(x=-(s['x'] * s['c']))

    with LogTrajectory(timespan) as lt:
        with solver():
            simulate(dynamics, sp0, timespan[0] - 1., timespan[-1] + 1.)

    correct = torch.tensor([9.0484, 8.1873, 7.4082, 6.7032, 6.0653, 5.4881,
                            4.9659, 4.4933, 4.0657, 3.6788], dtype=torch.float64)

    assert torch.allclose(lt.trajectory['x'], correct, atol=1e-4)


@pytest.mark.parametrize(
    "simulate_inner_bknd", [_diffeqdotjl_ode_simulate_inner, _torchdiffeq_ode_simulate_inner])
@pytest.mark.parametrize("x0", [torch.tensor(10.), torch.tensor([10., 5.])])
@pytest.mark.parametrize("c", [torch.tensor(0.1), torch.tensor([0.1, 0.2])])
def test_smoke_inner_simulates_forward_nd(simulate_inner_bknd, x0, c):
    sp0 = State(x=x0.double(), c=c.double())
    timespan = torch.linspace(0., 10., 10).double()

    simulate_inner_bknd(lambda s: State(x=-s['x'] * s['c']), sp0, timespan)


@pytest.mark.parametrize(
    "solver", [DiffEqDotJL, TorchDiffEq])
@pytest.mark.parametrize("x0", [torch.tensor(10.), torch.tensor([10., 5.])])
@pytest.mark.parametrize("c_", [torch.tensor(0.1), torch.tensor([0.1, 0.2])])
def test_gradcheck_inner_simulates_of_solvers_wrt_param(solver, x0, c_):
    c_ = c_.double().requires_grad_()
    timespan = torch.linspace(1.0, 10., 10).double()

    # TODO add other inputs we want to gradcheck.

    def dynamics(s: State):
        # FIXME
        #  -(s['x'] * s['c']) works but (-s['x']) * s['c'] errors on exceeding max ndim of 32. This occurs
        #  despite the fact that at every stage everything is a proper ndarray of dtype object.
        return State(x=-(s['x'] * s['c']))

    def wrapped_simulate(c):
        sp0 = State(x=x0.double(), c=c)
        with LogTrajectory(timespan) as lt:
            with solver():
                simulate(dynamics, sp0, timespan[0] - 1., timespan[-1] + 1.)
        return lt.trajectory['x']

    torch.autograd.gradcheck(wrapped_simulate, c_)


# TODO test whether float64 is properly required of initial state for diffeqpy backend.
# TODO test whether float64 is properly required of returned dstate even if initial state passes float64 check.
