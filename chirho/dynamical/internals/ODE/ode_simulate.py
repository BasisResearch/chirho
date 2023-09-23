from __future__ import annotations

import functools
from typing import TypeVar

from chirho.dynamical.internals.interruption import simulate_to_interruption
from chirho.dynamical.ops import State, simulate
from chirho.dynamical.ops.ODE import ODEBackend, ODEDynamics
from chirho.dynamical.ops.ODE.backends import TorchDiffEqBackend

S = TypeVar("S")
T = TypeVar("T")


@simulate.register(ODEDynamics)
def ode_simulate(
    dynamics: ODEDynamics,
    initial_state: State[T],
    timespan,
    *,
    backend: ODEBackend = TorchDiffEqBackend(),
    **kwargs,
):
    return _ode_simulate(backend, dynamics, initial_state, timespan, **kwargs)


@functools.singledispatch
def _ode_simulate(
    backend: ODEBackend,
    dynamics: ODEDynamics,
    initial_state: State[T],
    timespan,
    **kwargs,
):
    """
    Simulate an ODE dynamical system
    """
    raise NotImplementedError(
        f"ode_simulate not implemented for backend of type {type(backend)}"
    )


ode_simulate.register = _ode_simulate.register


@simulate_to_interruption.register(ODEDynamics)
def ode_simulate_to_interruption(
    dynamics: ODEDynamics,
    initial_state: State[T],
    timespan,
    *,
    backend: ODEBackend = TorchDiffEqBackend(),
    **kwargs,
):
    return _ode_simulate_to_interruption(
        backend, dynamics, initial_state, timespan, **kwargs
    )


@functools.singledispatch
def _ode_simulate_to_interruption(
    backend: ODEBackend,
    dynamics: ODEDynamics,
    initial_state: State[T],
    timespan,
    **kwargs,
):
    """
    Simulate an ODE dynamical system
    """
    raise NotImplementedError(
        f"ode_simulate_to_interruption not implemented for backend of type {type(backend)}"
    )


ode_simulate_to_interruption.register = _ode_simulate_to_interruption.register
