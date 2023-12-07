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

from .dynamical_fixtures import build_bayes_sir_dynamics

pyro.settings.set(module_local_params=True)

logger = logging.getLogger(__name__)

# Global variables for tests
init_state = State(S=torch.tensor(1.0), I=torch.tensor(2.0), R=torch.tensor(3.3))
start_time = torch.tensor(0.0)
end_time = torch.tensor(4.0)


def test_no_backend_error():
    sir = build_bayes_sir_dynamics()
    with pytest.raises(NotImplementedError):
        simulate(sir, init_state, start_time, end_time)


def test_backend_arg():
    sir = build_bayes_sir_dynamics()
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


# TODO this really is testing jit compilation AND gradcheck — split into two separate tests?
@pytest.mark.parametrize(
    # torchdiffeq needs torch versions of the last three dynfuncs, or to skip them.
    "solver", [DiffEqDotJL])  # , TorchDiffEq])
@pytest.mark.parametrize("x0", [torch.tensor(10.), torch.tensor([10., 5.])])
@pytest.mark.parametrize("c_", [torch.tensor(0.5), torch.tensor([0.5, 0.3])])
@pytest.mark.parametrize("dynfunc", [
    # Unsimplified defs just ensure the solutions are stable but still exercise the relevant ops. Julia does compile
    #  these away, but not until it's able to successfully push symbolics through the unsimplified python func.
    lambda s: -s['x'] * s['c'],
    lambda s: -(s['x'] * s['c']),
    lambda s: -.5 * s['x'] * s['c'],  # w/ python numeric type
    lambda s: -s['x'] / (1. / s['c']),
    lambda s: -s['x'] * (s['c'] + s['c'] - s['c']),
    lambda s: -s['x'] ** (s['c'] / s['c']),
    # The last three dynfuncs test numpy ufunc dispatch to julia.
    # .ravel and slice ensure that the return here always matches the initial state (the 2x2 matmul gives you 4 elems).
    lambda s: -((np.atleast_2d(s['x']).T @ np.atleast_2d(s['c'])) * s['x']).ravel()[:s['x'].size],
    lambda s: -(np.matmul(np.atleast_2d(s['x']).T, np.atleast_2d(s['c'])) * s['x']).ravel()[:s['x'].size],
    lambda s: np.sin(s['t']) + np.sin(np.pi + s['t']) - s['x'] * np.exp(np.log(s['c']))
])
def test_gradcheck_inner_simulates_of_solvers_wrt_param(solver, x0, c_, dynfunc):
    c_ = c_.double().requires_grad_()
    timespan = torch.linspace(1.0, 10., 10).double()

    # TODO gradcheck wrt time.

    def dynamics(s: State):
        dx = dynfunc(s)

        # for the test case where there are two parameters but only one (scalar) state variable.
        if x0.ndim == 0 and c_.ndim > 0:
            dx = dx.sum()

        return State(x=dx)

    def wrapped_simulate(c):
        sp0 = State(x=x0.double(), c=c)
        with LogTrajectory(timespan) as lt:
            simulate(dynamics, sp0, timespan[0] - 1., timespan[-1] + 1.)
        return lt.trajectory['x']

    # This goes outside the gradcheck b/c DiffEqDotJl lazily compiles the problem.
    with solver():
        torch.autograd.gradcheck(wrapped_simulate, c_, atol=1e-4)


# TODO test that the informative error fires when dstate returns something that isn't the right shape.
# TODO test that simulating with a dynamics different from that which a solver was compiled for fails.
# TODO test whether float64 is properly required of initial state for diffeqpy backend.
# TODO test whether float64 is properly required of returned dstate even if initial state passes float64 check.
