import warnings
from typing import Callable, Dict, Generic, Optional, Tuple, TypeVar, Union

import pyro
import torch

from chirho.dynamical.internals.interventional import intervene
from chirho.dynamical.ops.dynamical import ObservableInPlaceDynamics, State
from chirho.observational.handlers import condition

S = TypeVar("S")
T = TypeVar("T")


class Interruption(pyro.poutine.messenger.Messenger):
    # This is required so that the multiple inheritance works properly and super calls to this method execute the
    #  next implementation in the method resolution order.
    def _pyro_simulate_to_interruption(self, msg) -> None:
        pass


# TODO AZ - rename to static interruption?
class StaticInterruption(Interruption):
    def __init__(self, time: Union[float, torch.Tensor, T], **kwargs):
        self.time = torch.as_tensor(time)
        super().__init__(**kwargs)

    def _pyro_simulate_to_interruption(self, msg) -> None:
        _, _, _, start_time, end_time = msg["args"]

        if "next_static_interruption" in msg["kwargs"]:
            next_static_interruption = msg["kwargs"]["next_static_interruption"]
        else:
            next_static_interruption = None

        # If this interruption occurs within the timespan...
        if start_time < self.time < end_time:
            # Usurp the next static interruption if this one occurs earlier.
            if (
                next_static_interruption is None
                or self.time < next_static_interruption.time
            ):
                msg["kwargs"]["next_static_interruption"] = self
        elif self.time >= end_time:
            warnings.warn(
                f"{StaticInterruption.__name__} time {self.time} occurred after the end of the timespan "
                f"{end_time}. This interruption will have no effect.",
                UserWarning,
            )
        # Note AZiusld10 — This case is actually okay here within simulate_to_interruption, as calls may be specified
        # for the latter half of a particular timespan, and start only after some previous interruptions. So,
        # we put it in the simulate preprocess call instead, to get the full timespan. elif self.time < start_time:

        super()._pyro_simulate_to_interruption(msg)


class _InterventionMixin(Interruption):
    """
    We use this to provide the same functionality to both StaticIntervention and the DynamicIntervention,
     while allowing DynamicIntervention to not inherit StaticInterruption functionality.
    """

    def __init__(self, intervention: State[T], **kwargs):
        super().__init__(**kwargs)
        self.intervention = intervention

    def _pyro_apply_interruptions(self, msg) -> None:
        dynamics, initial_state = msg["args"]
        msg["args"] = (dynamics, intervene(initial_state, self.intervention))


class StaticIntervention(StaticInterruption, _InterventionMixin):
    """
    This effect handler interrupts a simulation at a given time, and
    applies an intervention to the state at that time.
    """

    pass


# Just a type rn for type checking. This should eventually have shared code in it useful for different types of point
#  observations.
class _PointObservationMixin:
    pass


class StaticObservation(Generic[T], StaticInterruption, _PointObservationMixin):
    def __init__(
        self,
        time: float,
        data: Dict[str, T],
        eps: float = 1e-6,
    ):
        self.data = data
        # Add a small amount of time to the observation time to ensure that
        # the observation occurs after the logging period.
        super().__init__(time + eps)

    def _pyro_apply_interruptions(self, msg) -> None:
        dynamics: ObservableInPlaceDynamics[T, None] = msg["args"][0]
        current_state: State[T] = msg["args"][1]

        with condition(data=self.data):
            with pyro.poutine.messenger.block_messengers(
                lambda m: isinstance(m, _PointObservationMixin) and (m is not self)
            ):
                dynamics.observation(current_state)

    def _pyro_sample(self, msg):
        # modify observed site names to handle multiple time points
        msg["name"] = msg["name"] + "_" + str(self.time.item())


class DynamicInterruption(Generic[T], Interruption):
    def __init__(
        self,
        event_f: Callable[[T, State[T]], T],
        var_order: Tuple[str, ...],
        max_applications: Optional[int] = None,
        **kwargs,
    ):
        """
        :param event_f: An event trigger function that approaches and returns 0.0 when the event should be triggered.
         This can be designed to trigger when the current state is "close enough" to some trigger state, or when an
         element of the state exceeds some threshold, etc. It takes both the current time and current state.
        :param var_order: The full State.var_order. This could be intervention.var_order if the intervention applies
         to the full state.
        :param max_applications: The maximum number of times this dynamic interruption can be applied. If None, there
            is no limit.
        """

        super().__init__(**kwargs)
        self.event_f = event_f
        self.var_order = var_order

        if max_applications is None or max_applications > 1:
            # This implies an infinite number of applications, but we don't support that yet, as we need some way
            #  of disabling a dynamic event proc for some time epsilon after it is triggered each time, otherwise
            #  it will just repeatedly trigger and the sim won't advance.
            raise NotImplementedError(
                "More than one application is not yet implemented."
            )

        self.max_applications = max_applications

    def _pyro_simulate_to_interruption(self, msg) -> None:
        if "dynamic_interruptions" not in msg["kwargs"]:
            msg["kwargs"]["dynamic_interruptions"] = []

        # Add self to the collection of dynamic interruptions.
        # TODO: This doesn't appear to be used anywhere. Is it needed?
        if self not in msg.get("ignored_dynamic_interruptions", []):
            msg["kwargs"]["dynamic_interruptions"].append(self)

        super()._pyro_simulate_to_interruption(msg)


class DynamicIntervention(DynamicInterruption, _InterventionMixin):
    """
    This effect handler interrupts a simulation when the given dynamic event function returns 0.0, and
    applies an intervention to the state at that time.
    """

    def __int__(
        self,
        intervention: State[T],
        event_f: Callable[[T, State[T]], T],
        var_order: Tuple[str, ...],
        max_applications: Optional[int] = None,
    ):
        """
        :param intervention: The instantaneous intervention applied to the state when the event is triggered.
        """

        super().__init__(
            event_f=event_f,
            var_order=var_order,
            max_applications=max_applications,
            intervention=intervention,
        )
