from __future__ import annotations

from typing import Generic, TypeVar

import pyro

T = TypeVar("T")


class TrajectoryLogging(Generic[T], pyro.poutine.messenger.Messenger):
    def __init__(self, tspan: T):
        self.tspan = tspan
        super().__init__()

    def _pyro_simulate(self, msg) -> None:
        pass
