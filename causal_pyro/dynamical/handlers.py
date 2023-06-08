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
)

import functools
import pyro
import torch
import torchdiffeq

from causal_pyro.dynamical.ops import (
    State,
    Dynamics,
    simulate,
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
        var_order: tuple[str, ...],
        time: torch.Tensor,
        state: tuple[T, ...],
    ) -> tuple[T, ...]:
        ddt, env = State(), State()
        for var, value in zip(var_order, state):
            setattr(env, var, value)
        dynamics.diff(ddt, env)
        return tuple(getattr(ddt, var, 0.0) for var in var_order)


@simulate.register(ODEDynamics)
@pyro.poutine.runtime.effectful(type="simulate")
def ode_simulate(
    dynamics: ODEDynamics, initial_state: State[torch.Tensor], timespan, **kwargs
):
    return ode_simulate_span(dynamics, initial_state, timespan, **kwargs)


@simulate_span.register(ODEDynamics)
@pyro.poutine.runtime.effectful(type="simulate_span")
def ode_simulate_span(dynamics: ODEDynamics, initial_state: State[torch.Tensor], timespan, **kwargs):
    var_order = tuple(sorted(initial_state.keys))  # arbitrary, but fixed

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


class SimulatorEventLoop(pyro.poutine.messenger.Messenger):

    def __enter__(self):
        return super().__enter__()

    def _pyro_simulate(self, msg) -> None:
        sorted_point_interruptions = sorted(msg["accrued_point_interruptions"], key=lambda x: x.time)

        # dynamic_interruptions = msg["accrued_dynamic_interruptions"]

        dynamics, current_state, full_timespan = msg["args"]
        
        span_trajectories = []

        prepended_sorted_point_interruptions = [None] + [*sorted_point_interruptions]
        lpspi = len(prepended_sorted_point_interruptions)

        for i in range(lpspi):

            curr_interruption = prepended_sorted_point_interruptions[i]

            span_start = curr_interruption.time if curr_interruption is not None else timespan[0]
            span_end = prepended_sorted_point_interruptions[i + 1].time if i < len(lpspi) - 1 else timespan[-1]

            timespan = torch.cat(
                (span_start[..., None],
                full_timespan[span_start < full_timespan < span_end],
                span_end[..., None]), dim=-1)

            with pyro.poutine.messenger.block_messengers(
                lambda m: isinstance(m, PointInterruption) and ((m is not curr_interruption) or (curr_interruption is None))):
                
                span_traj = simulate_span(dynamics, current_state, timespan)
                
            current_state = span_traj[-1]

            # FIXME this needs to slice the point interruption times away so the user gets what they asked for in the original tspan.
            span_trajectories.append(span_traj)
        # TODO: Raj implement me.
        full_trajectory = concatenate(span_trajectories)

        msg["value"] = full_trajectory
        msg["done"] = True


class DynamicInterruption(pyro.poutine.messenger.Messenger):
    # TODO AZ - I don't think this should subclass from PointInterruption, because
    #  it doesn't take a known time.
    #  Maybe we want a kind of noop abstraction though to make type checking easier?
    def __init__(self):
        raise NotImplementedError


class PointInterruption(pyro.poutine.messenger.Messenger):
    """
    This effect handler interrupts a simulation at a given time, and
    splits it into two separate calls to `simulate` with tspan1 = [tspan[0], time]
    and tspan2 = (time, tspan[-1]].
    """

    def __init__(self, time: Union[float, torch.Tensor], eps: float = 1e-10):
        self.time = torch.as_tensor(time)
        self.eps = eps

    def _pyro_simulate(self, msg) -> None:
        accrued_interruptions = msg.get("accrued_point_interruptions", [])
        accrued_interruptions.append(self)

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


class PointIntervention(PointInterruption):
    """
    This effect handler interrupts a simulation at a given time, and
    applies an intervention to the state at that time. The simulation
    is then split into two separate calls to `simulate` with tspan1 = [tspan[0], time]
    and tspan2 = (time, tspan[-1]].
    """

    def __init__(self, time: float, intervention: State[torch.Tensor], eps=1e-10):
        super().__init__(time, eps)
        self.intervention = intervention

    def _pyro_simulate_span(self, msg):
        # TODO needs to swap the starting state in args with an intervene starting state.
        raise NotImplementedError("Raj do this PLEASE")
        start_time = msg["kwargs"]["start_time"]
        end_time = msg["kwargs"]["end_time"]
        if start_time <= self.time < end_time:
            msg["kwargs"]["start_state"] = intervene(msg["kwargs"]["start_state"], self.intervention)


class PointObservation(PointInterruption):
    def __init__(
        self,
        time: float,
        loglikelihood: Callable[[State[torch.Tensor]], torch.Tensor],
        eps: float = 1e-10,
    ):
        super().__init__(time, eps)
        self.loglikelihood = loglikelihood

    def _pyro_post_simulate(self, msg) -> None:
        curr_state = msg["value"][-1]
        pyro.factor(f"obs_{self.time}", self.loglikelihood(curr_state))
        return super()._pyro_post_simulate(msg)
