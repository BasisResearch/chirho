import numbers
import typing
from typing import Callable, Dict, Generic, List, Optional, TypeVar, Union

import pyro
import torch

R = Union[numbers.Real, torch.Tensor]
S = TypeVar("S")
T = TypeVar("T")


class State(Generic[T], Dict[str, T]):
    pass


Dynamics = Callable[[State[T]], State[T]]


@pyro.poutine.runtime.effectful(type="simulate")
def simulate(
    dynamics: Dynamics[T],
    state: State[T],
    start_time: R,
    end_time: R,
    *,
    solver: Optional[S] = None,
    **kwargs,
) -> State[T]:
    """
    Simulate a dynamical system.
    """
    from chirho.dynamical.handlers.interruption import (
        DynamicInterruption,
        Interruption,
        StaticInterruption,
    )
    from chirho.dynamical.internals.solver import (
        Solver,
        get_dynamic_interruptions,
        get_next_interruptions,
        get_solver,
        get_static_interruptions,
        simulate_to_interruption,
    )

    solver_: Solver = get_solver() if solver is None else typing.cast(Solver, solver)

    static_interruptions: List[StaticInterruption] = get_static_interruptions()
    dynamic_interruptions: List[DynamicInterruption] = get_dynamic_interruptions()

    while start_time < end_time:
        terminal_interruptions, interruption_time = get_next_interruptions(
            solver_,
            dynamics,
            state,
            start_time,
            end_time,
            static_interruptions=static_interruptions,
            dynamic_interruptions=dynamic_interruptions,
        )

        state = simulate_to_interruption(
            solver_,
            dynamics,
            state,
            start_time,
            interruption_time,
        )

        start_time = interruption_time

        with pyro.poutine.messenger.block_messengers(
            lambda m: isinstance(m, Interruption) and m not in terminal_interruptions
        ):
            for interruption in terminal_interruptions:
                if isinstance(interruption, DynamicInterruption):
                    dynamic_interruptions.remove(interruption)
                if isinstance(interruption, StaticInterruption):
                    static_interruptions.remove(interruption)
                if hasattr(interruption, "apply"):
                    dynamics, state = interruption.apply(dynamics, state)
    return state
