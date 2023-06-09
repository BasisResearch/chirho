from typing import (
    Any,
    Callable,
    Generic,
    Hashable,
    Mapping,
    Optional,
    Tuple,
    TypeVar,
    Union,
    List
)

import functools
import pyro
import torch
import torchdiffeq
import warnings

from causal_pyro.dynamical.ops import (
    State,
    Dynamics,
    simulate,
    concatenate,
    simulate_span,
    Trajectory,
)
from causal_pyro.interventional.ops import intervene
from causal_pyro.interventional.handlers import do

S, T = TypeVar("S"), TypeVar("T")


class ODEDynamics(pyro.nn.PyroModule):
    def diff(self, dX: State[torch.Tensor], X: State[torch.Tensor]):
        raise NotImplementedError

    def forward(self, initial_state: State[torch.Tensor], timespan, **kwargs):
        return simulate(self, initial_state, timespan, **kwargs)

    def _deriv(
        dynamics: "ODEDynamics",
        var_order: Tuple[str, ...],
        time: torch.Tensor,
        state: Tuple[T, ...],
    ) -> Tuple[T, ...]:
        ddt, env = State(), State()
        for var, value in zip(var_order, state):
            setattr(env, var, value)
        dynamics.diff(ddt, env)
        return tuple(getattr(ddt, var, 0.0) for var in var_order)


@simulate.register(ODEDynamics)
@pyro.poutine.runtime.effectful(type="simulate")
def ode_simulate(
        dynamics: ODEDynamics, initial_state: State[torch.Tensor], timespan,
        event_fn: Optional[Callable[[torch.tensor, torch.tensor], torch.tensor]] = None, **kwargs
):
    return ode_simulate_span(dynamics, initial_state, timespan, event_fn=event_fn, **kwargs)


@simulate_span.register(ODEDynamics)
@pyro.poutine.runtime.effectful(type="simulate_span")
def ode_simulate_span(
        dynamics: ODEDynamics, initial_state: State[torch.Tensor], timespan,
        event_fn: Optional[Callable[[torch.tensor, torch.tensor], torch.tensor]] = None, **kwargs
):
    var_order = initial_state.var_order  # arbitrary, but fixed

    # # <DEBUG>
    # Apparently if your event never actually triggers, this just...runs forever. :unamused:
    # res = torchdiffeq.odeint_event(
    #     functools.partial(dynamics._deriv, var_order),
    #     tuple(getattr(initial_state, v) for v in var_order),
    #     timespan[0],
    #     event_fn=lambda t, s: t - torch.tensor(5.))
    # # </DEBUG>

    if event_fn is None:
        solns = torchdiffeq.odeint(
            functools.partial(dynamics._deriv, var_order),
            tuple(getattr(initial_state, v) for v in var_order),
            timespan,
            **kwargs,
        )
    else:
        # TODO this returns event time and final state...
        # odeint_event(func, y0, t0, *, event_fn, reverse_time=False, odeint_interface=odeint, **kwargs)
        raise NotImplementedError

    trajectory = Trajectory()
    for var, soln in zip(var_order, solns):
        setattr(trajectory, var, soln)

    return trajectory


class SimulatorEventLoop(pyro.poutine.messenger.Messenger):
    def __enter__(self):
        return super().__enter__()

    # TODO AZ make this partial/lazy so that the dynamic event funcs get defined and it only adds a new
    #  terminal condition event function for each simulate_step.
    def _build_combined_event_func(
            self, dynamic_interventions: List['DynamicIntervention'], terminal_t: torch.tensor) \
            -> Optional[Callable[[torch.tensor, torch.tensor], torch.tensor]]:
        """
        Builds a flattened dynamic event function that takes the time and a flattened state vector, and returns an
        array of values. If and only if an event should be triggered, the value at that index should be zero.
        :return: A callable that takes the time and a flattened state vector, and returns an array of values.
        """

        # Annoyingly, torchdiffeq's event function just runs forever if the terminal event never procs, so we just
        #  add another event that triggers when t exceeds terminal t.
        def terminal_event_func(t: torch.tensor, _) -> torch.tensor:
            # Return the time to terminal t if we haven't passed it, and otherwise, return 0.0.
            return torch.where(t < terminal_t, terminal_t - t, torch.tensor(0.0))

        def combined_event_func(t: torch.tensor, current_state_flat: torch.tensor):
            dynamic_intervention_event_fn_results = [
                di.flattened_event_f(t, current_state_flat) for di in dynamic_interventions]

            return torch.tensor(dynamic_intervention_event_fn_results + [terminal_event_func(t, current_state_flat)])

        return combined_event_func if len(dynamic_interventions) else None

    def _dynamic_pass_simulate_span(self, point_interruptions, dynamic_interruptions) \
            -> Tuple['PointIntervention', 'DynamicIntervention']:
        # TODO
        # Note: if there aren't any dynamic interruptions, just call simulate span.
        # 1. Build the event function with the terminal time event appended.
        # In a loop:
        # 2. Simulate the span with the event function.
        # 3. If it terminated on the terminal time, re-execute the

        # Nevermind. I think it's going to be easiest to do a full pass of the dynamic interruptions, and then
        #  splice them in as (point interruptions, dynamic interruptions). Where the actual point interruptions are
        #  just going to be (point interruptions, None). This just lets us block on the actual thing in the pyro stack.
        # Nope, that doesn't work b/c we need the point interruption states to apply...but all we want out of this is a
        #  the times of the dynamics, so that's far less bookkeeping that originally...

        pass

    def _pyro_simulate(self, msg) -> None:
        dynamics, initial_state, full_timespan = msg["args"]
        point_interruptions = sorted(
            msg.get("accrued_point_interruptions", []), key=lambda x: x.time
        )

        dynamic_interruptions = msg.get("accrued_dynamic_interruptions", [])

        # <TODO DELETE this sketch>
        # So for each simulate span, we need to exec _build_combined_event_func, with whatever the terminal
        #  time of that span is.
        # This needs to go into simulate span with the event_fn defined as the output of _build_combined_event_func.
        # This will include the terminal event where the tspan must stop.

        # We check first if the returned state gives us near zero for any of the event functions. If only the last one
        #  meets this criteria, assert as a sanity check that the returned time is close to the terminal time.
        # This tells us that no event proc'd, and we can run the span again without an event function in it.

        # If, however, an event does fire (some value is non-zero that isn't the last one). Then we need to execute
        #  the same logic as for the point interruptions, but terminating and intervening at the time the event procs.
        # So we "splice in" the pointified dynamic interruption, and then run the span again to get the proper splits.

        # The implementation below will break though because we can't insert into the list we're iterating over.
        # We may need to abstract components of that out to handle the case properly. The nice thing is that we don't
        #  actually need anything recursive. This is because the first pass already considers all events, so we know
        #  that none of the other events actually trigger in the time up to that event triggering.
        # <TODO DELETE>

        # <Error Check>
        # Check if the user specified two interventions at the same time.
        # Iterate all the adjacent pairs of the sorted point interruptions and check torch.isclose
        # on the time attribute.
        all_times = torch.tensor(
            [interruption.time for interruption in point_interruptions]
        )
        # Offset by one and hstack to get pairs.
        all_times = torch.stack([all_times[:-1], all_times[1:]], dim=-1)
        # Check if any of the pairs are close.
        if torch.any(torch.isclose(all_times[..., 0], all_times[..., 1])):
            raise ValueError("Two point interruptions cannot occur at the same time.")
        # </Error Check>

        # Simulate the span with no interruptions if there are none that should apply.
        if not len(point_interruptions):
            # TODO AZ-719odi1. See tag below. If we check whether the interruption time is within the timespan, maybe we
            #  don't need this particular block_messengers.
            with pyro.poutine.messenger.block_messengers(
                lambda m: isinstance(m, PointInterruption)
            ):
                msg["value"] = simulate_span(dynamics, initial_state, full_timespan)
                msg["done"] = True
                return

        # Handle initial timspan with no interruptions
        span_start = full_timespan[0]
        span_end = point_interruptions[0].time
        timespan = torch.cat(
            (
                span_start[..., None],
                full_timespan[
                    (span_start < full_timespan) & (full_timespan < span_end)
                ],
                span_end[..., None],
            ),
            dim=-1,
        )

        with pyro.poutine.messenger.block_messengers(
            lambda m: isinstance(m, PointInterruption)
        ):
            span_traj = simulate_span(dynamics, initial_state, timespan)

        current_state = span_traj[-1]
        # remove the point interruption time
        span_trajectories = [span_traj[:-1]]

        final_i = len(point_interruptions) - 1
        # Simulate between each point interruption
        for i, curr_interruption in enumerate(point_interruptions):
            span_start = curr_interruption.time
            span_end = (
                point_interruptions[i + 1].time if i < final_i else full_timespan[-1]
            )
            timespan = torch.cat(
                (
                    span_start[..., None],
                    full_timespan[
                        (span_start < full_timespan) & (full_timespan < span_end)
                    ],
                    span_end[..., None],
                ),
                dim=-1,
            )

            with pyro.poutine.messenger.block_messengers(
                lambda m: isinstance(m, PointInterruption)
                and (m is not curr_interruption)
            ):
                span_traj = simulate_span(dynamics, current_state, timespan)

            current_state = span_traj[-1]
            if i < final_i:
                span_traj = span_traj[1:-1]  # remove interruption times at endpoints
            else:
                span_traj = span_traj[1:]  # remove interruption time at start
            span_trajectories.append(span_traj)

        full_trajectory = concatenate(*span_trajectories)
        msg["value"] = full_trajectory
        msg["done"] = True


class PointInterruption(pyro.poutine.messenger.Messenger):
    """
    This effect handler interrupts a simulation at a given time, and
    splits it into two separate calls to `simulate` with tspan1 = [tspan[0], time]
    and tspan2 = (time, tspan[-1]].
    """

    def __init__(self, time: Union[float, torch.Tensor], eps: float = 1e-10, **kwargs):
        self.time = torch.as_tensor(time)
        self.eps = eps
        super().__init__(**kwargs)

    def _pyro_simulate(self, msg) -> None:
        dynamics, initial_state, full_timespan = msg["args"]
        if full_timespan[0] < self.time < full_timespan[-1]:
            accrued_interruptions = msg.get("accrued_point_interruptions", [])
            accrued_interruptions.append(self)
            msg["accrued_point_interruptions"] = accrued_interruptions

        # Throw an error if the intervention time occurs before the timespan, as the simulator
        #  won't simulate before the first time in the timespan, but the user might expect
        #  the intervention to affect the execution regardless.
        if self.time < full_timespan[0]:
            raise ValueError(
                f"Intervention time {self.time} is before the first time in the timespan {full_timespan[0]}. If you'd"
                f" like to include this intervention, explicitly include the earlier intervention time within the"
                f" range of the timespan."
            )

        # Throw a warning if the intervention time occurs after the timespan, as the user likely made a mistake
        #  on the intervention time.
        if self.time > full_timespan[-1]:
            warnings.warn(
                f"Intervention time {self.time} is after the last time in the timespan {full_timespan[-1]}."
                f" This intervention will have no effect on the simulation.",
                UserWarning,
            )

    # This isn't needed here, as the default PointInterruption doesn't need to
    #  affect the simulation of the span. This can be implemented in other
    #  handlers, however, see e.g. PointIntervention._pyro_simulate_span.
    # def _pyro_simulate_span(self, msg) -> None:


@intervene.register(State)
def state_intervene(obs: State[T], act: State[T], **kwargs) -> State[T]:
    new_state = State()
    for k in obs.keys:
        setattr(
            new_state, k, intervene(getattr(obs, k), getattr(act, k, None), **kwargs)
        )
    return new_state


class _InterventionMixin:
    """
    We use this to provide the same functionality to both PointIntervention and the DynamicIntervention,
     while allowing DynamicIntervention to not inherit PointInterruption functionality.
    """
    def __init__(self, intervention: State[torch.Tensor], **kwargs):
        super().__init__(**kwargs)
        self.intervention = intervention

    def _pyro_simulate_span(self, msg):
        """
        Replaces the current state (the state with which the span will begin simulation), with the intervened state,
         thereby making the span simulate from the intervened state rather than the original starting state.
        """
        # TODO AZ-719odi1. See tag above. We maybe do want to check here whether the intervention time is within the
        #  timespan.
        dynamics, current_state, full_timespan = msg["args"]
        intervened_state = intervene(current_state, self.intervention)
        msg["args"] = (dynamics, intervened_state, full_timespan)


class PointIntervention(PointInterruption, _InterventionMixin):
    """
    This effect handler interrupts a simulation at a given time, and
    applies an intervention to the state at that time. The simulation
    is then split into two separate calls to `simulate` with tspan1 = [tspan[0], time]
    and tspan2 = (time, tspan[-1]].
    """

    def __init__(self, time: float, intervention: State[torch.Tensor], eps=1e-10):
        super().__init__(time=time, eps=eps, intervention=intervention)


class PointObservation(PointInterruption):
    def __init__(
        self,
        time: float,
        loglikelihood: Callable[[State[torch.Tensor]], torch.Tensor],
        eps: float = 1e-10,
    ):
        super().__init__(time, eps)
        self.loglikelihood = loglikelihood

    def _pyro_simulate_span(self, msg) -> None:
        _, current_state, _ = msg["args"]
        pyro.factor(f"obs_{self.time}", self.loglikelihood(current_state))


# TODO AZ dji1820- I can't see any reason to abstract out a parent "DynamicInterruption" here?
class DynamicIntervention(pyro.poutine.messenger.Messenger, _InterventionMixin):
    def __init__(
            self,
            event_f: Callable[[torch.tensor, State[torch.tensor]], torch.tensor],
            intervention: State[torch.tensor],
            var_order: Tuple[str, ...]):

        """
        :param event_f: An event trigger function that approaches and returns 0.0 when the event should be triggered.
         This can be designed to trigger when the current state is "close enough" to some trigger state, or when an
         element of the state exceeds some threshold, etc. It takes both the current time and current state.
        :param intervention: The instantaneous intervention applied to the state when the event is triggered.
        :param var_order: The full State.var_order. This could be intervention.var_order if the intervention applies
         to the full state.
        """

        super().__init__(intervention=intervention)
        self.event_f = event_f
        self.var_order = var_order

    def as_point_intervention(self, time) -> PointIntervention:
        return PointIntervention(time=time, intervention=self.intervention)

    def flattened_event_f(self, t: torch.tensor, current_state_flat: torch.tensor):
        """
        This function is called by the solver that expects to pass
        :param t: The current time of the simulation.
        :param current_state_flat: The current state of the simulation, flattened into a single tensor.
        :return: A scalar that, when 0.0, tells the solver the event has been triggered.
        """

        # Reconstruct a state to pass to the user supplied event function.
        # TODO AZ — this should probably just be a factory method of State...
        current_state = State(**{k: v for k, v in zip(self.var_order, current_state_flat)})
        return self.event_f(t, current_state)

    def _pyro_simulate(self, msg):
        # Note dji1820: Despite not (currently) having an abstracted "DynamicInterruptions", am still calling these
        # "interruptions". That only means we wouldn't have to refactor to the more general term if we wanted to do
        # so later.
        accrued_dynamic_interruptions = msg.get("accrued_dynamic_interruptions", [])
        accrued_dynamic_interruptions.append(self)
        msg["accrued_dynamic_interruptions"] = accrued_dynamic_interruptions
