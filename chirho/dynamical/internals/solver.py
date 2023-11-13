from __future__ import annotations

import functools
import numbers
from typing import List, Optional, Tuple, TypeVar, Union

import pyro
import torch

from chirho.dynamical.internals._utils import ShallowMessenger
from chirho.dynamical.ops import Dynamics, State

R = Union[numbers.Real, torch.Tensor]
S = TypeVar("S")
T = TypeVar("T")


class Solver(pyro.poutine.messenger.Messenger):
    def _pyro_get_solver(self, msg) -> None:
        # Overwrite the solver in the message with the enclosing solver when used as a context manager.
        msg["value"] = self
        msg["done"] = True
        msg["stop"] = True


@pyro.poutine.runtime.effectful(type="get_solver")
def get_solver() -> Solver:
    """
    Get the current solver from the context.
    """
    raise ValueError("Solver not found in context.")


class Interruption(ShallowMessenger):
    def _pyro_get_new_interruptions(self, msg) -> None:
        if msg["value"] is None:
            msg["value"] = []
        assert isinstance(msg["value"], list)
        msg["value"].append(self)


@pyro.poutine.runtime.effectful(type="get_new_interruptions")
def get_new_interruptions() -> List[Interruption]:
    """
    Install the active interruptions into the context.
    """
    return []


@functools.singledispatch
def simulate_point(
    solver: Solver,
    dynamics: Dynamics[T],
    initial_state: State[T],
    start_time: R,
    end_time: R,
    **kwargs,
) -> State[T]:
    """
    Simulate a dynamical system.
    """
    raise NotImplementedError(
        f"simulate not implemented for solver of type {type(solver)}"
    )


@functools.singledispatch
def simulate_trajectory(
    solver: Solver,
    dynamics: Dynamics[T],
    initial_state: State[T],
    timespan: R,
    **kwargs,
) -> State[T]:
    """
    Simulate a dynamical system.
    """
    raise NotImplementedError(
        f"simulate_trajectory not implemented for solver of type {type(solver)}"
    )


# Separating out the effectful operation from the non-effectful dispatch on the default implementation
@pyro.poutine.runtime.effectful(type="simulate_to_interruption")
def simulate_to_interruption(
    interruption_stack: List[Interruption],
    dynamics: Dynamics[T],
    start_state: State[T],
    start_time: R,
    end_time: R,
    **kwargs,
) -> Tuple[State[T], R, Optional[Interruption]]:
    """
    Simulate a dynamical system until the next interruption.

    :returns: the final state
    """
    if len(interruption_stack) == 0:
        return simulate_point(get_solver(), dynamics, start_state, start_time, end_time, **kwargs), end_time, None

    raise NotImplementedError("No default behavior for simulate_to_interruption")


@pyro.poutine.runtime.effectful(type="apply_interruptions")
def apply_interruptions(
    dynamics: Dynamics[T], start_state: State[T]
) -> Tuple[Dynamics[T], State[T]]:
    """
    Apply the effects of an interruption to a dynamical system.
    """
    # Default is to do nothing.
    return dynamics, start_state


@functools.singledispatch
def get_next_interruptions(
    solver: Solver,
    dynamics: Dynamics[T],
    start_state: State[T],
    start_time: R,
    interruptions: List[Interruption],
) -> Tuple[Tuple[Interruption, ...], R]:
    raise NotImplementedError(
        f"get_next_interruptions not implemented for type {type(dynamics)}"
    )
