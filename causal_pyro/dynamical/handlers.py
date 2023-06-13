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
    simulate_to_interruption
)
from causal_pyro.interventional.ops import intervene
from causal_pyro.interventional.handlers import do

S, T = TypeVar("S"), TypeVar("T")


# noinspection PyPep8Naming
class ODEDynamics(pyro.nn.PyroModule):
    def diff(self, dX: State[torch.Tensor], X: State[torch.Tensor]):
        raise NotImplementedError

    def forward(self, initial_state: State[torch.Tensor], timespan, **kwargs):
        return simulate(self, initial_state, timespan, **kwargs)

    # noinspection PyMethodParameters
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
def torchdiffeq_simulate(
        dynamics: ODEDynamics, initial_state: State[torch.Tensor], timespan, **kwargs
):
    var_order = initial_state.var_order  # arbitrary, but fixed

    solns = torchdiffeq.odeint(
        functools.partial(dynamics._deriv, var_order),
        tuple(getattr(initial_state, v) for v in var_order),
        timespan,
        **kwargs,
    )

    trajectory = Trajectory()
    for var, soln in zip(var_order, solns):
        setattr(trajectory, var, soln)

    return trajectory


@simulate_to_interruption.register(ODEDynamics)
@pyro.poutine.runtime.effectful(type="simulate_to_interruption")
def torchdiffeq_ode_simulate_to_interruption(
        dynamics: ODEDynamics,
        start_state: State[torch.tensor],
        timespan,  # The first element of timespan is assumed to be the starting time.
        *,
        next_static_interruption: Optional['PointInterruption'] = None,
        dynamic_interruptions: Optional[List['DynamicInterruption']] = None, **kwargs
) -> Tuple[Trajectory[torch.tensor], Tuple['Interruption', ...], float, State[torch.tensor]]:

    nodyn = dynamic_interruptions is None or len(dynamic_interruptions) == 0
    nostat = next_static_interruption is None

    if nostat and nodyn:
        trajectory = torchdiffeq_simulate(dynamics, start_state, timespan, **kwargs)
        return trajectory, (), timespan[-1], trajectory[-1]

    elif nodyn:
        # TODO AZ just implementing this block for testing rn. Dynamic interruption will actually fold this stuff in.
        preinterrupt_tspan = torch.tensor(
            [*timespan[timespan < next_static_interruption.time], next_static_interruption.time])

        var_order = start_state.var_order  # arbitrary, but fixed

        solns = torchdiffeq.odeint(
            functools.partial(dynamics._deriv, var_order),
            tuple(getattr(start_state, v) for v in var_order),
            preinterrupt_tspan,
            **kwargs,
        )

        trajectory = Trajectory()
        for var, soln in zip(var_order, solns):
            setattr(trajectory, var, soln)

        return trajectory[:-1], (next_static_interruption,), next_static_interruption.time, trajectory[-1]

    else:

        # TODO this needs to
        # 1. Create the combined event function from the dynamic interruptions.
        # 2. Simulate out, with odeint_event, to either the end of the timespan or the static interruption.
        #   2.1 Do this by adding a terminal event function that triggers when terminal.
        # 3. If the end point fired, re-execute and return the tspan.
        # 4. If a dynamic interruption fired, re-run just that part and return the tspan.
        # 5. Also need to return something to let the caller know which event(s) was(/were) triggered so that the relevant
        #     executions and state/dynamics adjustments can happen.

        raise NotImplementedError("TODO")


class SimulatorEventLoop(pyro.poutine.messenger.Messenger):
    def __enter__(self):
        return super().__enter__()

    # TODO AZ 291okdu — move this out of here, maybe add a dispatcher for torch.tensor states and times?
    # def _build_combined_event_func(
    #         self, dynamic_interventions: List['DynamicIntervention'], terminal_t: torch.tensor) \
    #         -> Optional[Callable[[torch.tensor, torch.tensor], torch.tensor]]:
    #     """
    #     Builds a flattened dynamic event function that takes the time and a flattened state vector, and returns an
    #     array of values. If and only if an event should be triggered, the value at that index should be zero.
    #     :return: A callable that takes the time and a flattened state vector, and returns an array of values.
    #     """
    #
    #     # Annoyingly, torchdiffeq's event function just runs forever if the terminal event never procs, so we just
    #     #  add another event that triggers when t exceeds terminal t.
    #     def terminal_event_func(t: torch.tensor, _) -> torch.tensor:
    #         # Return the time to terminal t if we haven't passed it, and otherwise, return 0.0.
    #         return torch.where(t < terminal_t, terminal_t - t, torch.tensor(0.0))
    #
    #     def combined_event_func(t: torch.tensor, current_state_flat: torch.tensor):
    #         dynamic_intervention_event_fn_results = [
    #             di.flattened_event_f(t, current_state_flat) for di in dynamic_interventions]
    #
    #         return torch.tensor(dynamic_intervention_event_fn_results + [terminal_event_func(t, current_state_flat)])
    #
    #     return combined_event_func if len(dynamic_interventions) else None

    def _pyro_simulate(self, msg) -> None:

        dynamics, initial_state, full_timespan = msg["args"]

        # Initial values. These will be updated in the loop below.
        span_start_state = initial_state
        span_timespan = full_timespan

        # We use interruption mechanics to stop the timespan at the right point.
        default_terminal_interruption = PointInterruption(
            time=span_timespan[-1],
        )

        full_trajs = []
        first = True

        # Simulate through the timespan, stopping at each interruption. This gives e.g. intervention handlers
        #  a chance to modify the state and/or dynamics before the next span is simulated.
        while True:
            span_traj, terminal_interruptions, end_time, end_state = simulate_to_interruption(
                dynamics,
                span_start_state,
                span_timespan,
                # Here, we pass the default terminal interruption — the end of the timespan. Other point/static
                #  interruption handlers may replace this with themselves if they happen before the end.
                next_static_interruption=default_terminal_interruption,
                # We just pass nothing here, as any interruption handlers will be responsible for
                #  accruing themselves to the message. Leaving explicit for documentation.
                dynamic_interruptions=None
            )  # type: Trajectory[T], Tuple['Interruption', ...], float, State[T]

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
            span_timespan = torch.tensor([end_time, *full_timespan[full_timespan > end_time]])

            # Update the starting state.
            span_start_state = end_state

            first = False

        msg["value"] = concatenate(*full_trajs)
        msg["done"] = True


class Interruption(pyro.poutine.messenger.Messenger):
    def _pyro_simulate_to_interruption(self, msg) -> None:
        pass

    # TODO AZ 291okdu — move this out of here, maybe add a dispatcher for torch.tensor states and times?
    # def flattened_event_f(self, t: torch.tensor, current_state_flat: torch.tensor) -> torch.tensor:
    #     raise NotImplementedError("Inheritors must implement.")


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
            raise ValueError(f"{PointInterruption.__name__} time {self.time} occurred before the start of the "
                             f"timespan {start_time}. This interruption will have no effect.")

    def _pyro_simulate_to_interruption(self, msg) -> None:
        dynamics, initial_state, timespan = msg["args"]
        next_static_interruption = msg["kwargs"]["next_static_interruption"]

        start_time = timespan[0]
        end_time = timespan[-1]

        # If this interruption occurs within the timespan...
        if start_time < self.time < end_time:
            # Usurp the next static interruption if this one occurs earlier.
            if next_static_interruption is None or self.time < next_static_interruption.time:
                msg["kwargs"]["next_static_interruption"] = self
        elif self.time >= end_time:
            warnings.warn(f"{PointInterruption.__name__} time {self.time} occurred after the end of the timespan "
                          f"{end_time}. This interruption will have no effect.",
                          UserWarning)
        # Note AZiusld10 — This case is actually okay here, as simulate_to_interruption calls may be specified for
        # the latter half of a particular timespan, and start only after some previous interruptions. So,
        # we put it in the simulate preprocess call instead, to get the full timespan.
        # elif self.time < start_time:

        super()._pyro_simulate_to_interruption(msg)

    # TODO AZ 291okdu — move this out of here, maybe add a dispatcher for torch.tensor states and times?
    # def flattened_event_f(self, t: torch.tensor, current_state_flat: torch.tensor) -> torch.tensor:
    #     return torch.where(t < self.time, self.time - t, torch.tensor(0.0))


@intervene.register(State)
def state_intervene(obs: State[T], act: State[T], **kwargs) -> State[T]:
    new_state = State()
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

        self._last_interruptions = tuple()

    def _pyro_simulate_to_interruption(self, msg) -> None:

        # If this interruption was responsible for ending the previous simulation, apply its intervention.
        if self in self._last_interruptions:
            dynamics, initial_state, timespan = msg["args"]
            msg["args"] = (dynamics, intervene(initial_state, self.intervention), timespan)

        super()._pyro_simulate_to_interruption(msg)

    def _pyro_post_simulate_to_interruption(self, msg) -> None:
        # Parse out the returned tuple from the simulate_to_interruption call.
        _, triggered_interruptions, _, _ = msg["value"]

        # Record the collection of interruptions that stopped the last simulation.
        self._last_interruptions = triggered_interruptions


class PointIntervention(PointInterruption, _InterventionMixin):
    """
    This effect handler interrupts a simulation at a given time, and
    applies an intervention to the state at that time.
    """

    def __init__(self, time: float, intervention: State[torch.Tensor]):
        super().__init__(time=time, intervention=intervention)


class PointObservation(PointInterruption):
    def __init__(
        self,
        time: float,
        loglikelihood: Callable[[State[torch.Tensor]], torch.Tensor]
    ):
        super().__init__(time)
        self.loglikelihood = loglikelihood

    def _pyro_simulate_span(self, msg) -> None:
        _, current_state, _ = msg["args"]
        pyro.factor(f"obs_{self.time}", self.loglikelihood(current_state))


class DynamicInterruption(Interruption):
    def _pyro_simulate_to_interruption(self, msg) -> None:
        dynamic_interruptions = msg["kwargs"]["dynamic_interruptions"]

        if dynamic_interruptions is None:
            dynamic_interruptions = []

        # Add self to the collection of dynamic interruptions.
        msg["kwargs"]["dynamic_interruptions"] = dynamic_interruptions + [self]

        super()._pyro_simulate_to_interruption(msg)


class DynamicIntervention(DynamicInterruption, _InterventionMixin):
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

    # TODO AZ 291okdu — move this out of here, maybe add a dispatcher for torch.tensor states and times?
    #  Or do the factory suggestion below.
    # def flattened_event_f(self, t: torch.tensor, current_state_flat: torch.tensor):
    #     """
    #     This function is called by the solver that expects to pass
    #     :param t: The current time of the simulation.
    #     :param current_state_flat: The current state of the simulation, flattened into a single tensor.
    #     :return: A scalar that, when 0.0, tells the solver the event has been triggered.
    #     """
    #
    #     # Reconstruct a state to pass to the user supplied event function.
    #     # TODO AZ — this should probably just be a factory method of State...
    #     current_state = State(**{k: v for k, v in zip(self.var_order, current_state_flat)})
    #     return self.event_f(t, current_state)
