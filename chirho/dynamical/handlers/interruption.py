import bisect
import numbers
from typing import Callable, Generic, Tuple, TypeVar, Union

import pyro
import torch

from chirho.dynamical.handlers.trajectory import LogTrajectory
from chirho.dynamical.ops import Dynamics, State
from chirho.indexed.ops import get_index_plates, indices_of
from chirho.interventional.ops import Intervention, intervene
from chirho.observational.ops import Observation, observe

R = Union[numbers.Real, torch.Tensor]
S = TypeVar("S")
T = TypeVar("T")


class Interruption(pyro.poutine.messenger.Messenger):
    used: bool

    def __enter__(self):
        self.used = False
        return super().__enter__()

    def apply(self, dynamics: Dynamics, state: State[T]) -> Tuple[Dynamics, State[T]]:
        # Do nothing unless the interruption overwrites the apply method.
        return dynamics, state


class StaticInterruption(Interruption):
    time: R

    def __init__(self, time: R, *, eps: float = 1e-6):
        self.time = torch.as_tensor(time) + eps  # TODO enforce this where it is needed
        super().__init__()

    def _pyro_post_get_static_interruptions(self, msg) -> None:
        # static_interruptions are always sorted by time
        bisect.insort(msg["value"], self)

    def __lt__(self, other: "StaticInterruption"):
        return self.time < other.time


class DynamicInterruption(Generic[T], Interruption):
    """
    :param event_f: An event trigger function that approaches and returns 0.0 when the event should be triggered.
        This can be designed to trigger when the current state is "close enough" to some trigger state, or when an
        element of the state exceeds some threshold, etc. It takes both the current time and current state.
    """

    def __init__(self, event_f: Callable[[R, State[T]], R]):
        self.event_f = event_f
        super().__init__()

    def _pyro_post_get_dynamic_interruptions(self, msg) -> None:
        msg["value"].append(self)


class _InterventionMixin(Generic[T]):
    """
    We use this to provide the same functionality to both StaticIntervention and the DynamicIntervention,
     while allowing DynamicIntervention to not inherit StaticInterruption functionality.
    """

    intervention: Intervention[State[T]]

    def apply(self, dynamics: Dynamics, state: State[T]) -> Tuple[Dynamics, State[T]]:
        return dynamics, intervene(state, self.intervention)


class _PointObservationMixin(Generic[T]):
    observation: Observation[State[T]]
    time: R

    def apply(self, dynamics: Dynamics, state: State[T]) -> Tuple[Dynamics, State[T]]:
        return dynamics, observe(state, self.observation)

    def _pyro_sample(self, msg):
        # modify observed site names to handle multiple time points
        msg["name"] = msg["name"] + "_" + str(torch.as_tensor(self.time).item())


class StaticObservation(Generic[T], _PointObservationMixin[T], StaticInterruption):
    def __init__(
        self,
        time: R,
        observation: Observation[State[T]],
        *,
        eps: float = 1e-6,
    ):
        self.observation = observation
        # Add a small amount of time to the observation time to ensure that
        # the observation occurs after the logging period.
        super().__init__(time, eps=eps)


class StaticIntervention(Generic[T], _InterventionMixin[T], StaticInterruption):
    """
    This effect handler interrupts a simulation at a given time, and
    applies an intervention to the state at that time.

    :param time: The time at which the intervention is applied.
    :param intervention: The instantaneous intervention applied to the state when the event is triggered.
    """

    def __init__(
        self, time: R, intervention: Intervention[State[T]], *, eps: float = 1e-6
    ):
        self.intervention = intervention
        super().__init__(time, eps=eps)


class DynamicIntervention(Generic[T], DynamicInterruption, _InterventionMixin[T]):
    """
    This effect handler interrupts a simulation when the given dynamic event function returns 0.0, and
    applies an intervention to the state at that time.

    :param intervention: The instantaneous intervention applied to the state when the event is triggered.
    """

    def __init__(
        self,
        event_f: Callable[[R, State[T]], R],
        intervention: Intervention[State[T]],
    ):
        self.intervention = intervention
        super().__init__(event_f)


class StaticBatchObservation(Generic[T], LogTrajectory[T]):
    observation: Observation[State[T]]

    def __init__(
        self,
        times: torch.Tensor,
        observation: Observation[State[T]],
        *,
        eps: float = 1e-6,
    ):
        self.observation = observation
        super().__init__(times, eps=eps)

    def _pyro_post_simulate(self, msg) -> None:
        self.trajectory = observe(self.trajectory, self.observation)
