from __future__ import annotations

import functools
import numbers
import typing
from typing import List, Optional, Tuple, TypeVar, Union

import pyro
import torch

from chirho.dynamical.ops import Dynamics, State

if typing.TYPE_CHECKING:
    from chirho.dynamical.handlers.interruption import (
        DynamicInterruption,
        Interruption,
        StaticInterruption,
    )


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


@pyro.poutine.runtime.effectful(type="get_static_interruptions")
def get_static_interruptions() -> List[StaticInterruption]:
    """
    Get the current static interruptions from the context.
    """
    return []


@pyro.poutine.runtime.effectful(type="get_dynamic_interruptions")
def get_dynamic_interruptions() -> List[DynamicInterruption]:
    """
    Get the current dynamic interruptions from the context.
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
        f"simulate_point not implemented for solver of type {type(solver)}"
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
    solver: Solver,
    dynamics: Dynamics[T],
    start_state: State[T],
    start_time: R,
    end_time: R,
    **kwargs,
) -> State[T]:
    """
    Simulate a dynamical system until the next interruption. This will be either one of the passed
    dynamic interruptions, the next static interruption, or the end time, whichever comes first.

    :returns: the final state
    """
    return simulate_point(
        solver,
        dynamics,
        start_state,
        start_time,
        end_time,
    )


def get_next_interruptions(
    solver: Solver,
    dynamics: Dynamics[T],
    start_state: State[T],
    start_time: R,
    end_time: R,
    *,
    next_static_interruption: Optional[StaticInterruption] = None,
    dynamic_interruptions: List[DynamicInterruption] = [],
    **kwargs,
) -> Tuple[Tuple[Interruption, ...], R]:
    from chirho.dynamical.handlers.interruption import StaticInterruption

    if (
        isinstance(next_static_interruption, type(None))
        or typing.cast(StaticInterruption, next_static_interruption).time > end_time
    ):
        # If there's no static interruption or the next static interruption is after the end time,
        # we'll just simulate until the end time.
        next_static_interruption = StaticInterruption(time=end_time)

    assert isinstance(next_static_interruption, StaticInterruption)
    if len(dynamic_interruptions) == 0:
        # If there's no dynamic intervention, we'll simulate until either the end_time,
        # or the `next_static_interruption` whichever comes first.
        return (next_static_interruption,), next_static_interruption.time
    else:
        return get_next_interruptions_dynamic(
            solver,
            dynamics,
            start_state,
            start_time,
            next_static_interruption=next_static_interruption,
            dynamic_interruptions=dynamic_interruptions,
            **kwargs,
        )


@functools.singledispatch
def get_next_interruptions_dynamic(
    solver: Solver,
    dynamics: Dynamics[T],
    start_state: State[T],
    start_time: R,
    next_static_interruption: StaticInterruption,
    dynamic_interruptions: List[DynamicInterruption],
) -> Tuple[Tuple[Interruption, ...], R]:
    raise NotImplementedError(
        f"get_next_interruptions_dynamic not implemented for type {type(dynamics)}"
    )
