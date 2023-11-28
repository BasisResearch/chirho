from __future__ import annotations

import heapq
import math
import numbers
import typing
import warnings
from typing import Generic, List, Optional, Tuple, TypeVar, Union

import pyro
import torch

from chirho.dynamical.internals._utils import Prioritized, ShallowMessenger
from chirho.dynamical.ops import Dynamics, State

R = Union[numbers.Real, torch.Tensor]
S = TypeVar("S")
T = TypeVar("T")


class Solver(Generic[T], pyro.poutine.messenger.Messenger):
    @typing.final
    @staticmethod
    def _pyro_simulate(msg: dict) -> None:
        from chirho.dynamical.handlers.interruption import StaticInterruption

        dynamics: Dynamics[T] = msg["args"][0]
        state: State[T] = msg["args"][1]
        start_time: R = msg["args"][2]
        end_time: R = msg["args"][3]

        if pyro.settings.get("validate_dynamics"):
            check_dynamics(dynamics, state, start_time, end_time, **msg["kwargs"])

        # local state
        all_interruptions: List[Prioritized] = []
        heapq.heappush(
            all_interruptions,
            Prioritized(float(end_time), StaticInterruption(end_time)),
        )

        while start_time < end_time:
            for h in get_new_interruptions():
                if isinstance(h, StaticInterruption) and h.time >= end_time:
                    warnings.warn(
                        f"{StaticInterruption.__name__} {h} with time={h.time} "
                        f"occurred after the end of the timespan ({start_time}, {end_time})."
                        "This interruption will have no effect.",
                        UserWarning,
                    )
                elif isinstance(h, StaticInterruption) and h.time < start_time:
                    raise ValueError(
                        f"{StaticInterruption.__name__} {h} with time {h.time} "
                        f"occurred before the start of the timespan ({start_time}, {end_time})."
                        "This interruption will have no effect."
                    )
                else:
                    heapq.heappush(
                        all_interruptions,
                        Prioritized(float(getattr(h, "time", -math.inf)), h),
                    )

            possible_interruptions = []
            while all_interruptions:
                ph: Prioritized[Interruption] = heapq.heappop(all_interruptions)
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
                with next_interruption:
                    dynamics, state = apply_interruptions(dynamics, state)

                for h in possible_interruptions:
                    if h is not next_interruption:
                        heapq.heappush(
                            all_interruptions,
                            Prioritized(float(getattr(h, "time", -math.inf)), h),
                        )

        msg["value"] = state
        msg["done"] = True


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
