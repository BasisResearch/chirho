from __future__ import annotations

import functools
from typing import TypeVar

from chirho.dynamical.handlers.ODE import ODESolver
from chirho.dynamical.internals.interruption import simulate_to_interruption
from chirho.dynamical.ops.dynamical import State, simulate
from chirho.dynamical.ops.ODE import ODEDynamics

S = TypeVar("S")
T = TypeVar("T")


@simulate.register(ODEDynamics)
def ode_simulate(
    dynamics: ODEDynamics,
    initial_state: State[T],
    timespan,
    *,
    solver: ODESolver,
    **kwargs,
):
    return _ode_simulate(solver, dynamics, initial_state, timespan, **kwargs)


@functools.singledispatch
def _ode_simulate(
    solver: ODESolver,
    dynamics: ODEDynamics,
    initial_state: State[T],
    timespan,
    **kwargs,
):
    """
    Simulate an ODE dynamical system
    """
    raise NotImplementedError(
        f"ode_simulate not implemented for solver of type {type(solver)}"
    )


ode_simulate.register = _ode_simulate.register


@simulate_to_interruption.register(ODEDynamics)
def ode_simulate_to_interruption(
    dynamics: ODEDynamics,
    initial_state: State[T],
    timespan,
    *,
    solver: ODESolver,
    **kwargs,
):
    return _ode_simulate_to_interruption(
        solver, dynamics, initial_state, timespan, **kwargs
    )


@functools.singledispatch
def _ode_simulate_to_interruption(
    solver: ODESolver,
    dynamics: ODEDynamics,
    initial_state: State[T],
    timespan,
    **kwargs,
):
    """
    Simulate an ODE dynamical system
    """
    raise NotImplementedError(
        f"ode_simulate_to_interruption not implemented for solver of type {type(solver)}"
    )


ode_simulate_to_interruption.register = _ode_simulate_to_interruption.register
