from __future__ import annotations

import functools
from typing import Tuple, TypeVar

from chirho.dynamical.handlers.interruption import Interruption
from chirho.dynamical.handlers.ODE import ODESolver
from chirho.dynamical.internals.interruption import get_next_interruptions_dynamic
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


@get_next_interruptions_dynamic.register(ODEDynamics)
def ode_get_next_interruptions_dynamic(
    dynamics: ODEDynamics,
    initial_state: State[T],
    start_time: T,
    end_time: T,
    *,
    solver: ODESolver,
    **kwargs,
) -> Tuple[Tuple["Interruption", ...], T]:
    return _ode_get_next_interruptions_dynamic(
        solver, dynamics, initial_state, start_time, end_time, **kwargs
    )


# noinspection PyUnusedLocal
@functools.singledispatch
def _ode_get_next_interruptions_dynamic(
    solver: ODESolver,
    dynamics: ODEDynamics,
    initial_state: State[T],
    start_time: T,
    end_time: T,
    **kwargs,
) -> Tuple[Tuple["Interruption", ...], T]:
    """
    Simulate an ODE dynamical system
    """
    raise NotImplementedError(
        f"ode_get_next_interruptions_dynamic not implemented for solver of type {type(solver)}"
    )


ode_get_next_interruptions_dynamic.register = (
    _ode_get_next_interruptions_dynamic.register
)
