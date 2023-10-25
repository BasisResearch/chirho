import numbers
from typing import Callable, Generic, TypeVar, Union

import torch

from chirho.dynamical.handlers.trajectory import LogTrajectory
from chirho.dynamical.internals.solver import Interruption
from chirho.dynamical.ops import State
from chirho.indexed.ops import cond
from chirho.interventional.ops import Intervention, intervene
from chirho.observational.ops import Observation, observe

R = Union[numbers.Real, torch.Tensor]
S = TypeVar("S")
T = TypeVar("T")


class DependentInterruption(Generic[T], Interruption):
    event_f: Callable[[R, State[T]], R]


class StaticInterruption(Generic[T], DependentInterruption[T]):
    time: R

    def __init__(self, time: R):
        self.time = time
        super().__init__()

    def event_f(self, time: R, state: State[T]) -> R:
        return cond(0.0, self.time - time, case=time < self.time)


class DynamicInterruption(Generic[T], DependentInterruption[T]):
    """
    :param event_f: An event trigger function that approaches and returns 0.0 when the event should be triggered.
        This can be designed to trigger when the current state is "close enough" to some trigger state, or when an
        element of the state exceeds some threshold, etc. It takes both the current time and current state.
    """

    event_f: Callable[[R, State[T]], R]

    def __init__(self, event_f: Callable[[R, State[T]], R]):
        self.event_f = event_f
        super().__init__()


class _InterventionMixin(Generic[T]):
    """
    We use this to provide the same functionality to both StaticIntervention and the DynamicIntervention,
     while allowing DynamicIntervention to not inherit StaticInterruption functionality.
    """

    intervention: Intervention[State[T]]

    def _pyro_apply_interruptions(self, msg) -> None:
        dynamics, initial_state = msg["args"]
        msg["args"] = (dynamics, intervene(initial_state, self.intervention))


class _PointObservationMixin(Generic[T]):
    observation: Observation[State[T]]
    time: R

    def _pyro_apply_interruptions(self, msg) -> None:
        dynamics = msg["args"][0]
        state: State[T] = msg["args"][1]
        msg["value"] = (dynamics, observe(state, self.observation))

    def _pyro_sample(self, msg):
        # modify observed site names to handle multiple time points
        msg["name"] = msg["name"] + "_" + str(torch.as_tensor(self.time).item())


class StaticObservation(Generic[T], StaticInterruption[T], _PointObservationMixin[T]):
    def __init__(
        self,
        time: R,
        observation: Observation[State[T]],
    ):
        self.observation = observation
        # Add a small amount of time to the observation time to ensure that
        # the observation occurs after the logging period.
        super().__init__(time)


class StaticIntervention(Generic[T], StaticInterruption[T], _InterventionMixin[T]):
    """
    This effect handler interrupts a simulation at a given time, and
    applies an intervention to the state at that time.

    :param time: The time at which the intervention is applied.
    :param intervention: The instantaneous intervention applied to the state when the event is triggered.
    """

    def __init__(self, time: R, intervention: Intervention[State[T]]):
        self.intervention = intervention
        super().__init__(time)


class DynamicIntervention(Generic[T], DynamicInterruption[T], _InterventionMixin[T]):
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
    ):
        self.observation = observation
        super().__init__(times)

    def _pyro_post_simulate(self, msg) -> None:
        self.trajectory = observe(self.trajectory, self.observation)
