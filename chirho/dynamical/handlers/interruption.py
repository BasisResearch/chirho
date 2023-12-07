import numbers
import typing
from typing import Callable, Generic, Tuple, TypeVar, Union

import pyro
import pyro.contrib.autoname
import torch

from chirho.dynamical.handlers.trajectory import LogTrajectory
from chirho.dynamical.ops import Dynamics, State, on
from chirho.indexed.ops import cond
from chirho.interventional.ops import Intervention, intervene
from chirho.observational.ops import Observation, observe

R = Union[numbers.Real, torch.Tensor]
S = TypeVar("S")
T = TypeVar("T")


class ZeroEvent(Generic[T]):
    event_fn: Callable[[R, State[T]], R]

    def __init__(self, event_fn: Callable[[R, State[T]], R]):
        self.event_fn = event_fn
        super().__init__()

    def __call__(self, state: State[T]) -> bool:
        return bool(self.event_fn(typing.cast(torch.Tensor, state["t"]), state) == 0.0)


class StaticEvent(Generic[T], ZeroEvent[T]):
    time: torch.Tensor
    event_fn: Callable[[R, State[T]], R]

    def __init__(self, time: R):
        self.time = torch.as_tensor(time)
        super().__init__(
            lambda time, _: cond(0.0, self.time - time, case=time < self.time)
        )


def StaticInterruption(time: R):
    @on(StaticEvent(time))
    def callback(
        dynamics: Dynamics[T], state: State[T]
    ) -> Tuple[Dynamics[T], State[T]]:
        return dynamics, state

    return callback


def StaticObservation(time: R, observation: Observation[State[T]]):
    @on(StaticEvent(time))
    def callback(
        dynamics: Dynamics[T], state: State[T]
    ) -> Tuple[Dynamics[T], State[T]]:
        with pyro.contrib.autoname.scope(prefix=f"t={torch.as_tensor(time).item()}"):
            return dynamics, observe(state, observation)

    return callback


def StaticIntervention(time: R, intervention: Intervention[State[T]]):
    @on(StaticEvent(time))
    def callback(
        dynamics: Dynamics[T], state: State[T]
    ) -> Tuple[Dynamics[T], State[T]]:
        return dynamics, intervene(state, intervention)

    return callback


def DynamicInterruption(event_fn: Callable[[R, State[T]], R]):
    @on(ZeroEvent(event_fn))
    def callback(
        dynamics: Dynamics[T], state: State[T]
    ) -> Tuple[Dynamics[T], State[T]]:
        return dynamics, state

    return callback


def DynamicIntervention(
    event_fn: Callable[[R, State[T]], R], intervention: Intervention[State[T]]
):
    @on(ZeroEvent(event_fn))
    def callback(
        dynamics: Dynamics[T], state: State[T]
    ) -> Tuple[Dynamics[T], State[T]]:
        return dynamics, intervene(state, intervention)

    return callback


class StaticBatchObservation(Generic[T], LogTrajectory[T]):
    observation: Observation[State[T]]

    def __init__(
        self,
        times: torch.Tensor,
        observation: Observation[State[T]],
        **kwargs,
    ):
        self.observation = observation
        super().__init__(times, **kwargs)

    def _pyro_post_simulate(self, msg: dict) -> None:
        super()._pyro_post_simulate(msg)
        self.trajectory = observe(self.trajectory, self.observation)
