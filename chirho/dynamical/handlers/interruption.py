import numbers
import warnings
from typing import Callable, Generic, Optional, TypeVar, Union

import pyro
import torch

from chirho.dynamical.handlers.trajectory import LogTrajectory
from chirho.dynamical.ops import State
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

    def _pyro_get_next_interruptions(self, msg) -> None:
        raise NotImplementedError("shouldn't be here!")


class StaticInterruption(Interruption):
    time: R

    def __init__(self, time: R):
        self.time = torch.as_tensor(time)  # TODO enforce this where it is needed
        super().__init__()

    def _pyro_simulate(self, msg) -> None:
        _, _, start_time, end_time = msg["args"]

        if self.time < start_time:
            raise ValueError(
                f"{StaticInterruption.__name__} time {self.time} occurred before the start of the "
                f"timespan {start_time}. This interruption will have no effect."
            )
        elif self.time >= end_time:
            warnings.warn(
                f"{StaticInterruption.__name__} time {self.time} occurred after the end of the timespan "
                f"{end_time}. This interruption will have no effect.",
                UserWarning,
            )

    def _pyro_get_next_interruptions(self, msg) -> None:
        _, _, _, start_time, end_time = msg["args"]

        if start_time < self.time < end_time:
            next_static_interruption: Optional[StaticInterruption] = msg["kwargs"].get(
                "next_static_interruption", None
            )

            # Usurp the next static interruption if this one occurs earlier.
            if (
                next_static_interruption is None
                or self.time < next_static_interruption.time
            ):
                msg["kwargs"]["next_static_interruption"] = self


class DynamicInterruption(Generic[T], Interruption):
    """
    :param event_f: An event trigger function that approaches and returns 0.0 when the event should be triggered.
        This can be designed to trigger when the current state is "close enough" to some trigger state, or when an
        element of the state exceeds some threshold, etc. It takes both the current time and current state.
    """

    def __init__(self, event_f: Callable[[R, State[T]], R]):
        self.event_f = event_f
        super().__init__()

    def _pyro_get_next_interruptions(self, msg) -> None:
        msg["kwargs"].setdefault("dynamic_interruptions", []).append(self)


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


class StaticObservation(Generic[T], StaticInterruption, _PointObservationMixin[T]):
    def __init__(
        self,
        time: R,
        observation: Observation[State[T]],
    ):
        self.observation = observation
        # Add a small amount of time to the observation time to ensure that
        # the observation occurs after the logging period.
        super().__init__(time)


class StaticIntervention(Generic[T], StaticInterruption, _InterventionMixin[T]):
    """
    This effect handler interrupts a simulation at a given time, and
    applies an intervention to the state at that time.

    :param time: The time at which the intervention is applied.
    :param intervention: The instantaneous intervention applied to the state when the event is triggered.
    """

    def __init__(self, time: R, intervention: Intervention[State[T]]):
        self.intervention = intervention
        super().__init__(time)


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
    ):
        self.observation = observation
        super().__init__(times)

    def _pyro_post_simulate(self, msg) -> None:
        self.trajectory = observe(self.trajectory, self.observation)
