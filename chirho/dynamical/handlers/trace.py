from __future__ import annotations

from typing import Generic, TypeVar

import pyro

T = TypeVar("T")


class DynamicTrace(Generic[T], pyro.poutine.messenger.Messenger):
    def __init__(self, logging_times: T):
        self.logging_times = logging_times
        super().__init__()

    # TODO: Pick up here.
    def _pyro_simulate(self, msg) -> None:
        # Turn a simulate that returns a state into a simulate that returns a trajectory at each of the logging_times
        pass

    def _pyro_post_simulate(self, msg) -> None:
        # Concatenate all of the trajectories into a single trajectory
        pass
