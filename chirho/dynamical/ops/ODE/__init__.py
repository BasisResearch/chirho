from __future__ import annotations

from typing import Dict, Protocol, TypeVar, runtime_checkable

from chirho.dynamical.ops import Dynamics, State, simulate

S = TypeVar("S")
T = TypeVar("T")


@runtime_checkable
class ODEBackend(Protocol[T]):
    simulation_args: Dict[str, T]


# noinspection PyPep8Naming
class ODEDynamics(Dynamics[S, T]):
    def diff(self, dX: State[S], X: State[S]) -> T:
        raise NotImplementedError

    def observation(self, X: State[S]):
        raise NotImplementedError

    def forward(self, initial_state: State[S], timespan, **kwargs):
        return simulate(self, initial_state, timespan, **kwargs)
