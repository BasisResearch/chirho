from __future__ import annotations

import functools
import warnings
from typing import Callable, Dict, Generic, List, Optional, Tuple, TypeVar, Union

import pyro
import torch
import torchdiffeq

from chirho.dynamical.ops import (
    State,
    Trajectory,
    apply_interruptions,
    concatenate,
    simulate,
    simulate_to_interruption,
)
from chirho.indexed.ops import IndexSet, gather, indices_of, union
from chirho.interventional.handlers import intervene
from chirho.observational.handlers import condition

S = TypeVar("S")
T = TypeVar("T")


@indices_of.register
def _indices_of_state(state: State, *, event_dim: int = 0, **kwargs) -> IndexSet:
    return union(
        *(
            indices_of(getattr(state, k), event_dim=event_dim, **kwargs)
            for k in state.keys
        )
    )


@indices_of.register
def _indices_of_trajectory(
    trj: Trajectory, *, event_dim: int = 0, **kwargs
) -> IndexSet:
    return union(
        *(
            indices_of(getattr(trj, k), event_dim=event_dim + 1, **kwargs)
            for k in trj.keys
        )
    )


@gather.register(State)
def _gather_state(
    state: State[T], indices: IndexSet, *, event_dim: int = 0, **kwargs
) -> State[T]:
    return type(state)(
        **{
            k: gather(getattr(state, k), indices, event_dim=event_dim, **kwargs)
            for k in state.keys
        }
    )


@gather.register(Trajectory)
def _gather_trajectory(
    trj: Trajectory[T], indices: IndexSet, *, event_dim: int = 0, **kwargs
) -> Trajectory[T]:
    return type(trj)(
        **{
            k: gather(getattr(trj, k), indices, event_dim=event_dim + 1, **kwargs)
            for k in trj.keys
        }
    )



class SimulatorEventLoop(Generic[T], pyro.poutine.messenger.Messenger):
    def __enter__(self):
        return super().__enter__()

    # noinspection PyMethodMayBeStatic
    def _pyro_simulate(self, msg) -> None:
        dynamics, initial_state, full_timespan = msg["args"]

        # Initial values. These will be updated in the loop below.
        span_start_state = initial_state
        span_timespan = full_timespan

        # We use interruption mechanics to stop the timespan at the right point.
        default_terminal_interruption = PointInterruption(
            time=span_timespan[-1],
        )

        full_trajs = []  # type: List[Trajectory[T]]
        first = True

        last_terminal_interruptions = tuple()  # type: Tuple[Interruption, ...]
        interruption_counts = dict()  # type: Dict[Interruption, int]

        # Simulate through the timespan, stopping at each interruption. This gives e.g. intervention handlers
        #  a chance to modify the state and/or dynamics before the next span is simulated.
        while True:
            # Block any interruption's application that wouldn't be the result of an interruption that ended the last
            #  simulation.
            with pyro.poutine.messenger.block_messengers(
                lambda m: isinstance(m, Interruption)
                and m not in last_terminal_interruptions
            ):
                dynamics, span_start_state = apply_interruptions(
                    dynamics, span_start_state
                )

            # Block dynamic interventions that have triggered and applied more than the specified number of times.
            # This will prevent them from percolating up to the simulate_to_interruption execution.
            with pyro.poutine.messenger.block_messengers(
                lambda m: isinstance(m, DynamicInterruption)
                and m.max_applications <= interruption_counts.get(m, 0)
            ):
                (
                    span_traj,
                    terminal_interruptions,
                    end_time,
                    end_state,
                ) = simulate_to_interruption(  # This call gets handled by interruption handlers.
                    dynamics,
                    span_start_state,
                    span_timespan,
                    # Here, we pass the default terminal interruption — the end of the timespan. Other point/static
                    #  interruption handlers may replace this with themselves if they happen before the end.
                    next_static_interruption=default_terminal_interruption,
                    # We just pass nothing here, as any interruption handlers will be responsible for
                    #  accruing themselves to the message. Leaving explicit for documentation.
                    dynamic_interruptions=None,
                )  # type: Trajectory[T], Tuple['Interruption', ...], torch.Tensor, State[T]

            if len(terminal_interruptions) > 1:
                warnings.warn(
                    "Multiple events fired simultaneously. This results in undefined behavior.",
                    UserWarning,
                )

            for interruption in terminal_interruptions:
                interruption_counts[interruption] = (
                    interruption_counts.get(interruption, 0) + 1
                )

            last = default_terminal_interruption in terminal_interruptions

            # Update the full trajectory.
            if first:
                full_trajs.append(span_traj)
            else:
                # Hack off the end time of the previous simulate_to_interruption, as the user didn't request this.
                # if any(s == 0 for k in span_traj.keys for s in getattr(span_traj[..., 1:], k).shape):
                #     full_trajs.append(span_traj[..., 1:])
                # TODO support event_dim > 0
                span_traj_: Trajectory[T] = span_traj[..., 1:]
                full_trajs.append(span_traj_)

            # If we've reached the end of the timespan, break.
            if last:
                # The end state in this case will be the final tspan requested by the user, so we need to include.
                # TODO support event_dim > 0
                full_trajs.append(end_state.trajectorify())
                break

            # Construct the next timespan so that we simulate from the prevous interruption time.
            # TODO AZ — we should be able to detect when this eps is too small, as it will repeatedly trigger
            #  the same event at the same time.
            span_timespan = torch.cat(
                (end_time.unsqueeze(0), full_timespan[full_timespan > end_time])
            )

            # Update the starting state.
            span_start_state = end_state

            # Use these to block interruption handlers that weren't responsible for the last interruption.
            last_terminal_interruptions = terminal_interruptions

            first = False

        msg["value"] = concatenate(*full_trajs)
        msg["done"] = True


class Interruption(pyro.poutine.messenger.Messenger):
    # This is required so that the multiple inheritance works properly and super calls to this method execute the
    #  next implementation in the method resolution order.
    def _pyro_simulate_to_interruption(self, msg) -> None:
        pass


# TODO AZ - rename to static interruption?
class PointInterruption(Interruption):
    def __init__(self, time: Union[float, torch.Tensor], **kwargs):
        self.time = torch.as_tensor(time)
        super().__init__(**kwargs)

    def _pyro_simulate(self, msg) -> None:
        dynamics, initial_state, timespan = msg["args"]
        start_time = timespan[0]

        # See note tagged AZiusld10 below.
        if self.time < start_time:
            raise ValueError(
                f"{PointInterruption.__name__} time {self.time} occurred before the start of the "
                f"timespan {start_time}. This interruption will have no effect."
            )

    def _pyro_simulate_to_interruption(self, msg) -> None:
        dynamics, initial_state, timespan = msg["args"]
        next_static_interruption = msg["kwargs"]["next_static_interruption"]

        start_time = timespan[0]
        end_time = timespan[-1]

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
                f"{PointInterruption.__name__} time {self.time} occurred after the end of the timespan "
                f"{end_time}. This interruption will have no effect.",
                UserWarning,
            )
        # Note AZiusld10 — This case is actually okay here within simulate_to_interruption, as calls may be specified
        # for the latter half of a particular timespan, and start only after some previous interruptions. So,
        # we put it in the simulate preprocess call instead, to get the full timespan. elif self.time < start_time:

        super()._pyro_simulate_to_interruption(msg)


@intervene.register(State)
def state_intervene(obs: State[T], act: State[T], **kwargs) -> State[T]:
    new_state: State[T] = State()
    for k in obs.keys:
        setattr(
            new_state, k, intervene(getattr(obs, k), getattr(act, k, None), **kwargs)
        )
    return new_state


class _InterventionMixin(Interruption):
    """
    We use this to provide the same functionality to both PointIntervention and the DynamicIntervention,
     while allowing DynamicIntervention to not inherit PointInterruption functionality.
    """

    def __init__(self, intervention: State[torch.Tensor], **kwargs):
        super().__init__(**kwargs)
        self.intervention = intervention

    def _pyro_apply_interruptions(self, msg) -> None:
        dynamics, initial_state = msg["args"]
        msg["args"] = (dynamics, intervene(initial_state, self.intervention))


class PointIntervention(PointInterruption, _InterventionMixin):
    """
    This effect handler interrupts a simulation at a given time, and
    applies an intervention to the state at that time.
    """

    pass


# Just a type rn for type checking. This should eventually have shared code in it useful for different types of point
#  observations.
class _PointObservationMixin:
    pass


class NonInterruptingPointObservationArray(
    pyro.poutine.messenger.Messenger, _PointObservationMixin
):
    def __init__(
        self,
        times: torch.Tensor,
        data: Dict[str, torch.Tensor],
        eps: float = 1e-6,
    ):
        self.data = data

        # Add a small amount of time to the observation time to ensure that
        # the observation occurs after the logging period.
        self.times = times + eps

        # Require that the times are sorted. This is required by the index masking we do below.
        # TODO AZ sort this here (and the data too) accordingly?
        if not torch.all(self.times[1:] > self.times[:-1]):
            raise ValueError("The passed times must be sorted.")

        self._insert_mask_key = f"{self.__class__.__name__}.insert_mask"

        # Require that each data element maps 1:1 with the times.
        if not all(len(v) == len(times) for v in data.values()):
            raise ValueError(
                f"Each data element must have the same length as the passed times. Got lengths "
                f"{[len(v) for v in data.values()]} for data elements {[k for k in data.keys()]}, but "
                f"expected length {len(times)}."
            )

        super().__init__()

    def _pyro_simulate(self, msg) -> None:
        if self._insert_mask_key in msg:
            # Just to avoid having to splice in multiple handlers. Also, this suggests the user is using this handler
            #  in a suboptimal way, as they could just put their data into a single handler and avoid the overhead
            #  of extra handlers in the stack.
            raise ValueError(  # TODO AZ - this shouldn't be a value error probably. Is there a pyro handler error?
                f"Cannot use {self.__class__.__name__} within another {self.__class__.__name__}."
            )

        dynamics, initial_state, timespan = msg["args"]

        # Concatenate the timespan and the observation times, then sort. TODO find a way to use the searchsorted
        #  result to avoid sorting again?
        new_timespan, sort_indices = torch.sort(torch.cat((timespan, self.times)))

        # Get the mask covering where the times were spliced in.
        insert_mask = sort_indices >= len(timespan)

        # Do a sanity check that the times were inserted in the right places.
        assert torch.allclose(new_timespan[insert_mask], self.times), (
            "Sanity check failed! Observation times not "
            "spliced into user provided timespan as expected."
        )

        msg["args"] = (dynamics, initial_state, new_timespan)
        msg[self._insert_mask_key] = insert_mask

    def _pyro_post_simulate(self, msg) -> None:
        dynamics, initial_state, timespan = msg["args"]
        full_traj = msg["value"]
        insert_mask = msg[self._insert_mask_key]

        # Do a sanity check that the times were inserted in the right places.
        assert torch.allclose(timespan[insert_mask], self.times), (
            "Sanity check failed! Observation times not "
            "spliced into user provided timespan as expected."
        )

        with condition(data=self.data):
            # This blocks the handler from being called again, as it is already in the stack.
            with pyro.poutine.messenger.block_messengers(
                lambda m: isinstance(m, _PointObservationMixin) and (m is not self)
            ):
                # with pyro.plate("__time_plate", size=int(insert_mask.sum()), dim=-1):
                dynamics.observation(full_traj[insert_mask])

        # Remove the elements of the trajectory at the inserted points.
        msg["value"] = full_traj[~insert_mask]


class PointObservation(PointInterruption, _PointObservationMixin):
    def __init__(
        self,
        time: float,
        data: Dict[str, torch.Tensor],
        eps: float = 1e-6,
    ):
        self.data = data
        # Add a small amount of time to the observation time to ensure that
        # the observation occurs after the logging period.
        super().__init__(time + eps)

    def _pyro_simulate(self, msg) -> None:
        # Raise an error if the observation time is close to the start of the timespan. This is a temporary measure
        #  until issues arising from this case are understood and adressed.

        dynamics, initial_state, timespan = msg["args"]

        if torch.isclose(self.time, timespan[0], atol=1e-3, rtol=1e-3):
            raise ValueError(
                f"{PointObservation.__name__} time {self.time} occurred at the start of the timespan {timespan[0]}. "
                f"This is not currently supported."
            )

        super()._pyro_simulate(msg)

    def _pyro_apply_interruptions(self, msg) -> None:
        dynamics, current_state = msg["args"]

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
        dynamic_interruptions = msg["kwargs"]["dynamic_interruptions"]

        if dynamic_interruptions is None:
            dynamic_interruptions = []

        msg["kwargs"]["dynamic_interruptions"] = dynamic_interruptions

        # Add self to the collection of dynamic interruptions.
        if self not in msg.get("ignored_dynamic_interruptions", []):
            dynamic_interruptions.append(self)

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
