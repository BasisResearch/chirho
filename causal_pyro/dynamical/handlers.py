from __future__ import annotations

import functools
import warnings
from typing import Callable, Dict, Generic, List, Optional, Tuple, TypeVar, Union

import pyro
import torch
import torchdiffeq

from causal_pyro.dynamical.ops import (
    State,
    Trajectory,
    apply_interruptions,
    concatenate,
    simulate,
    simulate_to_interruption,
)
from causal_pyro.interventional.handlers import intervene

S = TypeVar("S")
T = TypeVar("T")


# noinspection PyPep8Naming
class ODEDynamics(pyro.nn.PyroModule):
    def diff(self, dX: State[torch.Tensor], X: State[torch.Tensor]):
        raise NotImplementedError

    def observation(self, X: State[torch.Tensor]):
        raise NotImplementedError

    def forward(self, initial_state: State[torch.Tensor], timespan, **kwargs):
        return simulate(self, initial_state, timespan, **kwargs)

    # noinspection PyMethodParameters
    def _deriv(
        dynamics: "ODEDynamics",
        var_order: Tuple[str, ...],
        time: torch.Tensor,
        state: Tuple[torch.Tensor, ...],
    ) -> Tuple[torch.Tensor, ...]:
        ddt: State[torch.Tensor] = State()
        env: State[torch.Tensor] = State()
        for var, value in zip(var_order, state):
            setattr(env, var, value)
        dynamics.diff(ddt, env)
        return tuple(getattr(ddt, var, torch.tensor(0.0)) for var in var_order)


# <Torchdiffeq Implementations>


def _torchdiffeq_ode_simulate_inner(
    dynamics: ODEDynamics, initial_state: State[torch.Tensor], timespan, **kwargs
):
    var_order = initial_state.var_order  # arbitrary, but fixed

    solns = torchdiffeq.odeint(
        functools.partial(dynamics._deriv, var_order),
        tuple(getattr(initial_state, v) for v in var_order),
        timespan,
        **kwargs,
    )

    trajectory: Trajectory[torch.Tensor] = Trajectory()
    for var, soln in zip(var_order, solns):
        setattr(trajectory, var, soln)

    return trajectory


@simulate.register(ODEDynamics)
@pyro.poutine.runtime.effectful(type="simulate")
def torchdiffeq_ode_simulate(
    dynamics: ODEDynamics, initial_state: State[torch.Tensor], timespan, **kwargs
):
    return _torchdiffeq_ode_simulate_inner(dynamics, initial_state, timespan, **kwargs)


@simulate_to_interruption.register(ODEDynamics)
@pyro.poutine.runtime.effectful(type="simulate_to_interruption")
def torchdiffeq_ode_simulate_to_interruption(
    dynamics: ODEDynamics,
    start_state: State[torch.Tensor],
    timespan,  # The first element of timespan is assumed to be the starting time.
    *,
    next_static_interruption: Optional["PointInterruption"] = None,
    dynamic_interruptions: Optional[List["DynamicInterruption"]] = None,
    **kwargs,
) -> Tuple[
    Trajectory[torch.Tensor], Tuple["Interruption", ...], float, State[torch.Tensor]
]:
    nodyn = dynamic_interruptions is None or len(dynamic_interruptions) == 0
    nostat = next_static_interruption is None

    if nostat and nodyn:
        trajectory = _torchdiffeq_ode_simulate_inner(
            dynamics, start_state, timespan, **kwargs
        )
        return trajectory, (), timespan[-1], trajectory[-1]

    # Leaving these undone for now, just so we don't have to split test coverage. Once we get a better test suite
    #  for the many possibilities, this can be optimized.
    # TODO AZ if no dynamic events, just skip the event function pass.

    if dynamic_interruptions is None:
        dynamic_interruptions = []

    if nostat:
        # This is required because torchdiffeq.odeint_event appears to just go on and on forever without a terminal
        #  event.
        raise ValueError(
            "No static terminal interruption provided, but about to perform an event sim."
        )
    # for linter, because it's not deducing this from the if statement above.
    assert next_static_interruption is not None

    # Create the event function combining all dynamic events and the terminal (next) static interruption.
    combined_event_f = torchdiffeq_combined_event_f(
        next_static_interruption, dynamic_interruptions
    )

    # Simulate to the event execution.
    event_time, event_state = torchdiffeq.odeint_event(
        functools.partial(dynamics._deriv, start_state.var_order),
        tuple(getattr(start_state, v) for v in start_state.var_order),
        timespan[0],
        event_fn=combined_event_f,
    )

    # event_state has both the first and final state of the interrupted simulation. We just want the last.
    event_state = tuple(s[-1] for s in event_state)

    # Check which event(s) fired, and put the triggered events in a list.
    fired_mask = torch.isclose(
        combined_event_f(event_time, event_state),
        torch.tensor(0.0),
        rtol=1e-02,
        atol=1e-03,
    )

    if not torch.any(fired_mask):
        # TODO AZ figure out the tolerance of the odeint_event function and use that above.
        raise RuntimeError(
            "The solve terminated but no element of the event function output was within "
            "tolerance of zero."
        )

    if len(fired_mask) != len(dynamic_interruptions) + 1:
        raise RuntimeError(
            "The event function returned an unexpected number of events."
        )

    triggered_events = [
        de for de, fm in zip(dynamic_interruptions, fired_mask[:-1]) if fm
    ]
    if fired_mask[-1]:
        triggered_events.append(next_static_interruption)

    # Construct a new timespan that cuts off measurements after the event fires, but that includes the event time.
    timespan_2nd_pass = torch.tensor([*timespan[timespan < event_time], event_time])

    # Execute a standard, non-event based simulation on the new timespan.
    trajectory = _torchdiffeq_ode_simulate_inner(
        dynamics, start_state, timespan_2nd_pass, **kwargs
    )

    # Return that trajectory (with interruption time separated out into the end state), the list of triggered
    #  events, the time of the triggered event, and the state at the time of the triggered event.
    return trajectory[:-1], tuple(triggered_events), event_time, trajectory[-1]


# TODO AZ — maybe to multiple dispatch on the interruption type and state type?
def torchdiffeq_point_interruption_flattened_event_f(
    pi: "PointInterruption",
) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
    """
    Construct a flattened event function for a point interruption.
    :param pi: The point interruption for which to build the event function.
    :return: The constructed event function.
    """

    def event_f(t: torch.Tensor, _):
        return torch.where(t < pi.time, pi.time - t, torch.tensor(0.0))

    return event_f


# TODO AZ — maybe do multiple dispatch on the interruption type and state type?
def torchdiffeq_dynamic_interruption_flattened_event_f(
    di: "DynamicInterruption",
) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
    """
    Construct a flattened event function for a dynamic interruption.
    :param di: The dynamic interruption for which to build the event function.
    :return: The constructed event function.
    """

    def event_f(t: torch.Tensor, flat_state: torch.Tensor):
        # Torchdiffeq operates over flattened state tensors, so we need to unflatten the state to pass it the
        #  user-provided event function of time and State.
        state: State[torch.Tensor] = State(
            **{k: v for k, v in zip(di.var_order, flat_state)}
        )
        return di.event_f(t, state)

    return event_f


# TODO AZ — maybe do multiple dispatch on the interruption type and state type?
def torchdiffeq_combined_event_f(
    next_static_interruption: "PointInterruption",
    dynamic_interruptions: List["DynamicInterruption"],
) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
    """
    Construct a combined event function from a list of dynamic interruptions and a single terminal static interruption.
    :param next_static_interruption: The next static interruption. Viewed as terminal in the context of this event func.
    :param dynamic_interruptions: The dynamic interruptions.
    :return: The combined event function, taking in state and time, and returning a vector of floats. When any element
     of this vector is zero, the corresponding event terminates the simulation.
    """
    terminal_event_f = torchdiffeq_point_interruption_flattened_event_f(
        next_static_interruption
    )
    dynamic_event_fs = [
        torchdiffeq_dynamic_interruption_flattened_event_f(di)
        for di in dynamic_interruptions
    ]

    def combined_event_f(t: torch.Tensor, flat_state: torch.Tensor):
        return torch.tensor(
            [
                *[f(t, flat_state) for f in dynamic_event_fs],
                terminal_event_f(t, flat_state),
            ]
        )

    return combined_event_f


# <Torchdiffeq Implementations>


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
                )  # type: Trajectory[T], Tuple['Interruption', ...], float, State[T]

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
                full_trajs.append(span_traj[1:])

            # If we've reached the end of the timespan, break.
            if last:
                # The end state in this case will be the final tspan requested by the user, so we need to include.
                full_trajs.append(end_state.trajectorify())
                break

            # Construct the next timespan so that we simulate from the prevous interruption time.
            # TODO AZ — we should be able to detect when this eps is too small, as it will repeatedly trigger
            #  the same event at the same time.
            span_timespan = torch.tensor(
                [end_time, *full_timespan[full_timespan > end_time]]
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


class NonInterruptingPointObservationArray(pyro.poutine.messenger.Messenger, _PointObservationMixin):

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
            raise ValueError(
                f"The passed times must be sorted."
            )

        self._insert_mask_key = f"{self.__class__.__name__}.insert_mask"

        # Require that each data element maps 1:1 with the times.
        if not all(len(v) == len(times) for v in data.values()):
            raise ValueError(
                f"Each data element must have the same length as the passed times. Got lengths "
                f"{[len(v) for v in data.values()]} for data elements {[k for k in data.keys()]}, but "
                f"expected length {len(times)}."
            )

        super().__init__()

    def _pyro_sample(self, msg) -> None:
        # This tells pyro that the sample statement needs broadcasting.
        msg['fn'] = msg['fn'].to_event(1)

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
        assert torch.allclose(new_timespan[insert_mask], self.times), "Sanity check failed! Observation times not " \
                                                                      "spliced into user provided timespan as expected."

        msg["args"] = (dynamics, initial_state, new_timespan)
        msg[self._insert_mask_key] = insert_mask

    def _pyro_post_simulate(self, msg) -> None:
        dynamics, initial_state, timespan = msg["args"]
        full_traj = msg["value"]
        insert_mask = msg[self._insert_mask_key]

        # Do a sanity check that the times were inserted in the right places.
        assert torch.allclose(timespan[insert_mask], self.times), "Sanity check failed! Observation times not " \
                                                                  "spliced into user provided timespan as expected."

        with pyro.condition(data=self.data):
            # This blocks the handler from being called again, as it is already in the stack.
            with pyro.poutine.messenger.block_messengers(
                lambda m: isinstance(m, _PointObservationMixin) and (m is not self)
            ):
                dynamics.observation(full_traj[insert_mask])

        # Remove the elements of the trajectory at the inserted points.
        msg["value"] = full_traj[~insert_mask]


# TODO AZ - pull out common stuff between this and the interrupting observation, and have interrupting and non
#  inherit from the same.
# FIXME this needs to catch conflicting time observations.
class NonInterruptingPointObservation(pyro.poutine.messenger.Messenger, _PointObservationMixin):
    def __init__(
        self,
        time: float,
        data: Dict[str, torch.Tensor],
        eps: float = 1e-6,
    ):
        self.data = data

        # Add a small amount of time to the observation time to ensure that
        # the observation occurs after the logging period.
        self.time = torch.as_tensor(time + eps)

        self.time_str = str(self.time.item())

        super().__init__()

    # TODO make generic and not use tensor directly?
    def _pyro_simulate(self, msg) -> None:
        dynamics, initial_state, timespan = msg["args"]

        # Splice the time into the measured timespan, and get the corresponding index.

        insert_idx = torch.searchsorted(timespan, self.time)

        new_timespan = torch.cat(
            [timespan[:insert_idx], torch.tensor([self.time]), timespan[insert_idx:]]
        )

        msg["args"] = (dynamics, initial_state, new_timespan)
        msg[f"{NonInterruptingPointObservation.__name__}.inserted_idx_{self.time_str}"] = insert_idx

    def _pyro_post_simulate(self, msg) -> None:

        dynamics, initial_state, timespan = msg["args"]
        full_traj = msg["value"]

        # Observe the state at the inserted index.
        inserted_idx = msg[f"{NonInterruptingPointObservation.__name__}.inserted_idx_{self.time_str}"]

        with pyro.condition(data=self.data):
            # This blocks sample statements of other observation handlers.
            with pyro.poutine.messenger.block_messengers(
                lambda m: isinstance(m, _PointObservationMixin) and (m is not self)
            ):
                dynamics.observation(full_traj[inserted_idx.item()])

        # Remove the inserted index from the returned trajectory so the user won't see it.
        msg["value"] = concatenate(
            full_traj[:inserted_idx], full_traj[inserted_idx + 1:]
        )

    def _pyro_sample(self, msg):
        # modify observed site names to handle multiple time points
        msg["name"] = msg["name"] + "_" + self.time_str


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

        with pyro.condition(data=self.data):
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
