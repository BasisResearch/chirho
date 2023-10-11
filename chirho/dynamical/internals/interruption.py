from __future__ import annotations

import functools
import numbers
from typing import TYPE_CHECKING, List, Optional, Tuple, TypeVar, Union

import pyro
import torch

from chirho.dynamical.handlers.interruption.interruption import (
    DynamicInterruption,
    Interruption,
    StaticInterruption,
)
from chirho.dynamical.ops.dynamical import InPlaceDynamics, State, simulate

if TYPE_CHECKING:
    from chirho.dynamical.handlers.solver import Solver


R = Union[numbers.Real, torch.Tensor]
S = TypeVar("S")
T = TypeVar("T")


# Separating out the effectful operation from the non-effectful dispatch on the default implementation
@pyro.poutine.runtime.effectful(type="simulate_to_interruption")
def simulate_to_interruption(
    solver: "Solver",  # Quoted type necessary w/ TYPE_CHECKING to avoid circular import error
    dynamics: InPlaceDynamics[T],
    start_state: State[T],
    start_time: R,
    end_time: R,
    *,
    next_static_interruption: Optional[StaticInterruption] = None,
    dynamic_interruptions: List[DynamicInterruption] = [],
    **kwargs,
) -> Tuple[State[T], Tuple[Interruption, ...], R]:
    """
    Simulate a dynamical system until the next interruption. This will be either one of the passed
    dynamic interruptions, the next static interruption, or the end time, whichever comes first.
    :returns: the final state, a collection of interruptions that ended the simulation
    (this will usually just be a single interruption), and the time the interruption occurred.
    """

    interruptions, interruption_time = get_next_interruptions(
        solver,
        dynamics,
        start_state,
        start_time,
        end_time,
        next_static_interruption=next_static_interruption,
        dynamic_interruptions=dynamic_interruptions,
        **kwargs,
    )
    # TODO: consider memoizing results of `get_next_interruptions` to avoid recomputing
    #  the solver in the dynamic setting. The interactions are a bit tricky here though, as we couldn't be in
    #  a DynamicTrace context.
    event_state = simulate(
        dynamics, start_state, start_time, interruption_time, solver=solver
    )

    return event_state, interruptions, interruption_time


def get_next_interruptions(
    solver: "Solver",  # Quoted type necessary w/ TYPE_CHECKING to avoid circular import error
    dynamics: InPlaceDynamics[T],
    start_state: State[T],
    start_time: R,
    end_time: R,
    *,
    next_static_interruption: Optional[StaticInterruption] = None,
    dynamic_interruptions: List[DynamicInterruption] = [],
    **kwargs,
) -> Tuple[Tuple[Interruption, ...], R]:
    nodyn = len(dynamic_interruptions) == 0
    nostat = next_static_interruption is None

    if nostat or next_static_interruption.time > end_time:  # type: ignore
        # If there's no static interruption or the next static interruption is after the end time,
        # we'll just simulate until the end time.
        next_static_interruption = StaticInterruption(time=end_time)

    assert isinstance(
        next_static_interruption, StaticInterruption
    )  # Linter needs a hint

    if nodyn:
        # If there's no dynamic intervention, we'll simulate until either the end_time,
        # or the `next_static_interruption` whichever comes first.
        return (next_static_interruption,), next_static_interruption.time  # type: ignore
    else:
        return get_next_interruptions_dynamic(  # type: ignore
            solver,  # type: ignore
            dynamics,  # type: ignore
            start_state,  # type: ignore
            start_time,  # type: ignore
            next_static_interruption=next_static_interruption,
            dynamic_interruptions=dynamic_interruptions,
            **kwargs,
        )

    raise ValueError("Unreachable code reached.")


# noinspection PyUnusedLocal
@functools.singledispatch
def get_next_interruptions_dynamic(
    solver: "Solver",  # Quoted type necessary w/ TYPE_CHECKING to avoid circular import error
    dynamics: InPlaceDynamics[T],
    start_state: State[T],
    start_time: R,
    next_static_interruption: StaticInterruption,
    dynamic_interruptions: List[DynamicInterruption],
) -> Tuple[Tuple[Interruption, ...], R]:
    raise NotImplementedError(
        f"get_next_interruptions_dynamic not implemented for type {type(dynamics)}"
    )


@pyro.poutine.runtime.effectful(type="apply_interruptions")
def apply_interruptions(
    dynamics: InPlaceDynamics[T], start_state: State[T]
) -> Tuple[InPlaceDynamics[T], State[T]]:
    """
    Apply the effects of an interruption to a dynamical system.
    """
    # Default is to do nothing.
    return dynamics, start_state
