from __future__ import annotations

import functools
from typing import List, Optional, Tuple, TypeVar

from chirho.dynamical.handlers.interruption import (
    DynamicInterruption,
    Interruption,
    StaticInterruption,
)
from chirho.dynamical.handlers.ODE import ODESolver
from chirho.dynamical.internals.dynamical import simulate_trajectory
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


@simulate_trajectory.register(ODEDynamics)
def ode_simulate_trajectory(
    dynamics: ODEDynamics,
    initial_state: State[T],
    timespan: T,
    *,
    solver: ODESolver,
    **kwargs,
) -> State[T]:
    return _ode_simulate_trajectory(solver, dynamics, initial_state, timespan, **kwargs)


# noinspection PyUnusedLocal
@functools.singledispatch
def _ode_simulate_trajectory(
    solver: ODESolver,
    dynamics: ODEDynamics,
    initial_state: State[T],
    timespan: T,
    **kwargs,
):
    raise NotImplementedError(
        f"ode_simulate_trajectory not implemented for solver of type {type(solver)}"
    )


@get_next_interruptions_dynamic.register(ODEDynamics)
def ode_get_next_interruptions_dynamic(
    dynamics: ODEDynamics[S, T],
    start_state: State[T],
    start_time: T,
    next_static_interruption: "StaticInterruption",
    dynamic_interruptions: List["DynamicInterruption"],
    *,
    solver: Optional[ODESolver] = None,
    **kwargs,
) -> Tuple[Tuple["Interruption", ...], T]:
    return _ode_get_next_interruptions_dynamic(
        solver,
        dynamics,
        start_state,
        start_time,
        next_static_interruption,
        dynamic_interruptions,
        **kwargs,
    )


# noinspection PyUnusedLocal
@functools.singledispatch
def _ode_get_next_interruptions_dynamic(
    solver: Optional[ODESolver],
    dynamics: ODEDynamics[S, T],
    start_state: State[T],
    start_time: T,
    next_static_interruption: StaticInterruption,
    dynamic_interruptions: List[DynamicInterruption],
    **kwargs,
) -> Tuple[Tuple[Interruption, ...], T]:
    """
    Simulate an ODE dynamical system
    """
    raise NotImplementedError(
        f"ode_get_next_interruptions_dynamic not implemented for solver of type {type(solver)}"
    )
