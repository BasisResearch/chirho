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
def ode_simulate_span(
    dynamics: ODEDynamics, initial_state: State[torch.Tensor], timespan, **kwargs
):
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
        dynamics, initial_state, full_timespan = msg["args"]
        point_interruptions = sorted(
            msg.get("accrued_point_interruptions", []), key=lambda x: x.time
        )

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

        if not len(point_interruptions):
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
        dynamics, current_state, full_timespan = msg["args"]
        intervened_state = intervene(current_state, self.intervention)
        msg["args"] = (dynamics, intervened_state, full_timespan)


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
