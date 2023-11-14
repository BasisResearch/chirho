from __future__ import annotations

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
    pass


class Interruption(ShallowMessenger):
    def _pyro_get_new_interruptions(self, msg) -> None:
        if msg["value"] is None:
            msg["value"] = []
        assert isinstance(msg["value"], list)
        msg["value"].append(self)


class SolverRuntimeCheckHandler(pyro.poutine.messenger.Messenger):
    pass


@functools.singledispatch
def get_solver_runtime_check_handler(solver: Solver) -> SolverRuntimeCheckHandler:
    """
    Get the runtime check handler for the solver.
    """
    raise NotImplementedError(
        f"get_solver_runtime_check_handler not implemented for type {type(solver)}"
    )

 
@pyro.poutine.runtime.effectful(type="get_new_interruptions")
def get_new_interruptions() -> List[Interruption]:
    """
    Install the active interruptions into the context.
    """
    return []


@pyro.poutine.runtime.effectful(type="simulate_point")
def simulate_point(
    dynamics: Dynamics[T],
    initial_state: State[T],
    start_time: R,
    end_time: R,
    **kwargs,
) -> State[T]:
    """
    Simulate a dynamical system.
    """
    raise NotImplementedError("No default behavior for simulate_point")


@pyro.poutine.runtime.effectful(type="simulate_trajectory")
def simulate_trajectory(
    dynamics: Dynamics[T],
    initial_state: State[T],
    timespan: R,
    **kwargs,
) -> State[T]:
    """
    Simulate a dynamical system.
    """
    raise NotImplementedError("No default behavior for simulate_trajectory")


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
        return (
            simulate_point(dynamics, start_state, start_time, end_time, **kwargs),
            end_time,
            None,
        )

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
