from __future__ import annotations

import contextlib
import heapq
import math
import typing
import warnings
from typing import Generic, List, TypeVar

import pyro

from chirho.dynamical.handlers.interruption import StaticInterruption
from chirho.dynamical.internals._utils import Prioritized
from chirho.dynamical.internals.solver import (
    Interruption,
    apply_interruptions,
    get_new_interruptions,
    simulate_to_interruption,
)

S = TypeVar("S")
T = TypeVar("T")


class InterruptionEventLoop(Generic[T], pyro.poutine.messenger.Messenger):
    @typing.final
    @staticmethod
    def _pyro_simulate(msg) -> None:
        dynamics, state, start_time, end_time = msg["args"]

        # local state
        all_interruptions: List[Prioritized] = []
        heapq.heappush(
            all_interruptions,
            Prioritized(float(end_time), StaticInterruption(end_time)),
        )

        with msg["kwargs"].pop("solver", contextlib.nullcontext()):
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
