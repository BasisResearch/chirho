import numbers
from typing import Callable, Generic, Tuple, TypeVar, Union

import pyro
import pyro.contrib.autoname
import torch

from chirho.dynamical.handlers.trajectory import LogTrajectory
from chirho.dynamical.internals.solver import Interruption
from chirho.dynamical.ops import Dynamics, State
from chirho.indexed.ops import cond
from chirho.interventional.ops import Intervention, intervene
from chirho.observational.ops import Observation, observe

R = Union[numbers.Real, torch.Tensor]
S = TypeVar("S")
T = TypeVar("T")


class DependentInterruption(Generic[T], Interruption[T]):
    event_fn: Callable[[R, State[T]], R]

    def apply_fn(
        self, dynamics: Dynamics[T], state: State[T]
    ) -> Tuple[Dynamics[T], State[T]]:
        return dynamics, state


class StaticInterruption(Generic[T], DependentInterruption[T]):
    time: torch.Tensor

    def __init__(self, time: R):
        self.time = torch.as_tensor(time)
        super().__init__()

    def event_fn(self, time: R, state: State[T]) -> R:
        return cond(0.0, self.time - time, case=time < self.time)


class DynamicInterruption(Generic[T], DependentInterruption[T]):
    """
    :param event_f: An event trigger function that approaches and returns 0.0 when the event should be triggered.
        This can be designed to trigger when the current state is "close enough" to some trigger state, or when an
        element of the state exceeds some threshold, etc. It takes both the current time and current state.
    """

    event_fn: Callable[[R, State[T]], R]

    def __init__(self, event_fn: Callable[[R, State[T]], R]):
        self.event_fn = event_fn
        super().__init__()


class StaticObservation(Generic[T], StaticInterruption[T]):
    observation: Observation[State[T]]
    time: torch.Tensor

    def __init__(
        self,
        time: R,
        observation: Observation[State[T]],
    ):
        self.observation = observation
        # Add a small amount of time to the observation time to ensure that
        # the observation occurs after the logging period.
        super().__init__(time)

    def apply_fn(
        self, dynamics: Dynamics[T], state: State[T]
    ) -> Tuple[Dynamics[T], State[T]]:
        with pyro.contrib.autoname.scope(
            prefix=f"t={torch.as_tensor(self.time).item()}"
        ):
            return dynamics, observe(state, self.observation)


class StaticIntervention(Generic[T], StaticInterruption[T]):
    """
    This effect handler interrupts a simulation at a given time, and
    applies an intervention to the state at that time.

    :param time: The time at which the intervention is applied.
    :param intervention: The instantaneous intervention applied to the state when the event is triggered.
    """

    intervention: Intervention[State[T]]

    def __init__(self, time: R, intervention: Intervention[State[T]]):
        self.intervention = intervention
        super().__init__(time)

    def apply_fn(
        self, dynamics: Dynamics[T], state: State[T]
    ) -> Tuple[Dynamics[T], State[T]]:
        return dynamics, intervene(state, self.intervention)


class DynamicIntervention(Generic[T], DynamicInterruption[T]):
    """
    This effect handler interrupts a simulation when the given dynamic event function returns 0.0, and
    applies an intervention to the state at that time.

    :param intervention: The instantaneous intervention applied to the state when the event is triggered.
    """

    intervention: Intervention[State[T]]

    def __init__(
        self,
        event_fn: Callable[[R, State[T]], R],
        intervention: Intervention[State[T]],
    ):
        self.intervention = intervention
        super().__init__(event_fn)

    def apply_fn(
        self, dynamics: Dynamics[T], state: State[T]
    ) -> Tuple[Dynamics[T], State[T]]:
        return dynamics, intervene(state, self.intervention)


class StaticBatchObservation(Generic[T], LogTrajectory[T]):
    observation: Observation[State[T]]

    def __init__(
        self,
        times: torch.Tensor,
        observation: Observation[State[T]],
    ):
        self.observation = observation
        super().__init__(times)

    def _pyro_simulate(self, msg: dict) -> None:
        # We use a continuation to ensure that the observation is applied to the trajectory after the simulation
        #  has completed, regardless of the order of the handlers.
        def obs_continuation(msg: dict) -> None:
            self.trajectory = observe(self.trajectory, self.observation)

        if msg["continuation"] is None:
            msg["continuation"] = obs_continuation
        else:
            msg["continuation"] = lambda new_msg: obs_continuation(
                msg["continuation"](new_msg)
            )

        super()._pyro_simulate(msg)
