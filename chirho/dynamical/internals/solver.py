from __future__ import annotations

import heapq
import math
import numbers
import typing
import warnings
from typing import Callable, Generic, List, Optional, Tuple, TypeVar, Union

import pyro
import torch

from chirho.dynamical.internals._utils import Prioritized, ShallowMessenger
from chirho.dynamical.ops import Dynamics, State, on

R = Union[numbers.Real, torch.Tensor]
S = TypeVar("S")
T = TypeVar("T")


class Interruption(Generic[T], ShallowMessenger):
    predicate: Callable[[State[T]], bool]
    callback: Callable[[Dynamics[T], State[T]], Tuple[Dynamics[T], State[T]]]

    def __init__(
        self,
        predicate: Callable[[State[T]], bool],
        callback: Callable[[Dynamics[T], State[T]], Tuple[Dynamics[T], State[T]]],
    ):
        self.predicate = predicate
        self.callback = callback

    def _pyro_get_new_interruptions(self, msg: dict) -> None:
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


class Solver(Generic[T], pyro.poutine.messenger.Messenger):
    @staticmethod
    def _prioritize_interruption(h: Interruption[T]) -> Prioritized[Interruption[T]]:
        from chirho.dynamical.handlers.interruption import StaticEvent, ZeroEvent

        if isinstance(h.predicate, StaticEvent):
            return Prioritized(float(h.predicate.time), h)
        elif isinstance(h.predicate, ZeroEvent):
            return Prioritized(-math.inf, h)
        else:
            raise NotImplementedError(f"cannot install interruption {h}")

    @typing.final
    def _pyro_simulate(self, msg: dict) -> None:
        from chirho.dynamical.handlers.interruption import StaticEvent

        dynamics: Dynamics[T] = msg["args"][0]
        state: State[T] = msg["args"][1]
        start_time: R = msg["args"][2]
        end_time: R = msg["args"][3]

        if pyro.settings.get("validate_dynamics"):
            check_dynamics(dynamics, state, start_time, end_time, **msg["kwargs"])

        # local state
        all_interruptions: List[Prioritized[Interruption[T]]] = []
        heapq.heappush(
            all_interruptions,
            self._prioritize_interruption(
                on(StaticEvent(end_time), lambda d, s: (d, s))
            ),
        )

        while start_time < end_time:
            for h in get_new_interruptions():
                if isinstance(h.predicate, StaticEvent) and h.predicate.time > end_time:
                    warnings.warn(
                        f"{Interruption.__name__} {h} with time={h.predicate.time} "
                        f"occurred after the end of the timespan ({start_time}, {end_time})."
                        "This interruption will have no effect.",
                        UserWarning,
                    )
                elif (
                    isinstance(h.predicate, StaticEvent)
                    and h.predicate.time < start_time
                ):
                    raise ValueError(
                        f"{Interruption.__name__} {h} with time {h.predicate.time} "
                        f"occurred before the start of the timespan ({start_time}, {end_time})."
                        "This interruption will have no effect."
                    )
                else:
                    heapq.heappush(all_interruptions, self._prioritize_interruption(h))

            possible_interruptions: List[Interruption[T]] = []
            while all_interruptions:
                ph: Prioritized[Interruption[T]] = heapq.heappop(all_interruptions)
                possible_interruptions.append(ph.item)
                if ph.priority > start_time:
                    break

            state, start_time, next_interruption = simulate_to_interruption(
                possible_interruptions,
                dynamics,
                state,
                start_time,
                end_time,
                **msg["kwargs"],
            )

            if next_interruption is not None:
                dynamics, state = next_interruption.callback(dynamics, state)

                for h in possible_interruptions:
                    if h is not next_interruption:
                        heapq.heappush(
                            all_interruptions, self._prioritize_interruption(h)
                        )

        msg["value"] = state
        msg["done"] = True


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
    interruption_stack: List[Interruption[T]],
    dynamics: Dynamics[T],
    start_state: State[T],
    start_time: R,
    end_time: R,
    **kwargs,
) -> Tuple[State[T], R, Optional[Interruption[T]]]:
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


@pyro.poutine.runtime.effectful(type="check_dynamics")
def check_dynamics(
    dynamics: Dynamics[T],
    initial_state: State[T],
    start_time: R,
    end_time: R,
    **kwargs,
) -> None:
    """
    Validate a dynamical system.
    """
    pass


DYNAMICS_VALIDATION_ENABLED: bool = False


@pyro.settings.register("validate_dynamics", __name__, "DYNAMICS_VALIDATION_ENABLED")
def _check_validate_dynamics_flag(value: bool) -> None:
    assert isinstance(value, bool)
