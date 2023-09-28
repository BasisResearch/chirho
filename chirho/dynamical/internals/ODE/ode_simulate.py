from __future__ import annotations

import functools
from typing import TypeVar

from chirho.dynamical.handlers.ODE import ODESolver
from chirho.dynamical.internals.interruption import get_next_interruptions
from chirho.dynamical.ops.dynamical import State, simulate
from chirho.dynamical.ops.ODE import ODEDynamics

S = TypeVar("S")
T = TypeVar("T")


@simulate.register(ODEDynamics)
def ode_simulate(
    dynamics: ODEDynamics,
    initial_state: State[T],
    start_time: T,
    end_time: T,
    *,
    solver: ODESolver,
    **kwargs,
) -> State[T]:
    return _ode_simulate(
        solver, dynamics, initial_state, start_time, end_time, **kwargs
    )


# noinspection PyUnusedLocal
@functools.singledispatch
def _ode_simulate(
    solver: ODESolver,
    dynamics: ODEDynamics,
    initial_state: State[T],
    start_time: T,
    end_time: T,
    **kwargs,
) -> State[T]:
    """
    Simulate an ODE dynamical system
    """
    raise NotImplementedError(
        f"ode_simulate not implemented for solver of type {type(solver)}"
    )


ode_simulate.register = _ode_simulate.register


@get_next_interruptions.register(ODEDynamics)
def ode_get_next_interruptions(
    dynamics: ODEDynamics,
    initial_state: State[T],
    start_time,
    end_time,
    *,
    solver: ODESolver,
    **kwargs,
):
    return _ode_get_next_interruptions(
        solver, dynamics, initial_state, start_time, end_time, **kwargs
    )


# noinspection PyUnusedLocal
@functools.singledispatch
def _ode_get_next_interruptions(
    solver: ODESolver,
    dynamics: ODEDynamics,
    initial_state: State[T],
    start_time,
    end_time,
    **kwargs,
):
    """
    Simulate an ODE dynamical system
    """
    raise NotImplementedError(
        f"ode_get_next_interruptions not implemented for solver of type {type(solver)}"
    )


ode_get_next_interruptions.register = _ode_get_next_interruptions.register
