from __future__ import annotations

import functools
from typing import TYPE_CHECKING, Callable, List, Optional, Tuple, TypeVar

import pyro
import torch
import torchdiffeq

from chirho.dynamical.internals.interruption import simulate_to_interruption
from chirho.dynamical.ops import State, Trajectory, simulate
from chirho.dynamical.ops.ODE import ODEDynamics

if TYPE_CHECKING:
    from chirho.dynamical.internals.interruption import (
        DynamicInterruption,
        Interruption,
        PointInterruption,
    )

S = TypeVar("S")
T = TypeVar("T")


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


# TODO - determine if this is needed.
# @pyro.nn.pyro_method
@pyro.poutine.runtime.effectful(type="simulate")
def _torchdiffeq_ode_simulate_inner(
    dynamics: ODEDynamics, initial_state: State[torch.Tensor], timespan, **kwargs
):
    var_order = initial_state.var_order  # arbitrary, but fixed

    solns = _batched_odeint(  # torchdiffeq.odeint(
        functools.partial(_deriv, dynamics, var_order),
        tuple(getattr(initial_state, v) for v in var_order),
        timespan,
        **kwargs,
    )

    trajectory: Trajectory[torch.Tensor] = Trajectory()
    for var, soln in zip(var_order, solns):
        setattr(trajectory, var, soln)

    return trajectory


def _batched_odeint(
    func: Callable[[torch.Tensor, Tuple[torch.Tensor, ...]], Tuple[torch.Tensor, ...]],
    y0: Tuple[torch.Tensor, ...],
    t: torch.Tensor,
    *,
    event_fn=None,
    **kwargs,
) -> Tuple[torch.Tensor, ...]:
    """
    Vectorized torchdiffeq.odeint.
    """
    # TODO support event_dim > 0
    event_dim = 0  # assume states are batches of values of rank event_dim

    y0_batch_shape = torch.broadcast_shapes(
        *(y0_.shape[: len(y0_.shape) - event_dim] for y0_ in y0)
    )

    y0_expanded = tuple(
        # y0_[(None,) * (len(y0_batch_shape) - (len(y0_.shape) - event_dim)) + (...,)]
        y0_.expand(y0_batch_shape + y0_.shape[len(y0_.shape) - event_dim :])
        for y0_ in y0
    )

    if event_fn is not None:
        event_t, yt_raw = torchdiffeq.odeint_event(
            func, y0_expanded, t, event_fn=event_fn, **kwargs
        )
    else:
        yt_raw = torchdiffeq.odeint(func, y0_expanded, t, **kwargs)

    yt = tuple(
        torch.transpose(
            yt_[(..., None) + yt_.shape[len(yt_.shape) - event_dim :]],
            -len(yt_.shape) - 1,
            -1 - event_dim,
        )[0]
        for yt_ in yt_raw
    )
    return yt if event_fn is None else (event_t, yt)


@simulate.register(ODEDynamics)
def torchdiffeq_ode_simulate(
    dynamics: ODEDynamics, initial_state: State[torch.Tensor], timespan, **kwargs
):
    return _torchdiffeq_ode_simulate_inner(dynamics, initial_state, timespan, **kwargs)


@simulate_to_interruption.register(ODEDynamics)
@pyro.poutine.runtime.effectful(type="simulate_to_interruption")
@pyro.poutine.block(hide_types=["simulate"])
def torchdiffeq_ode_simulate_to_interruption(
    dynamics: ODEDynamics,
    start_state: State[torch.Tensor],
    timespan,  # The first element of timespan is assumed to be the starting time.
    *,
    next_static_interruption: Optional["PointInterruption"] = None,
    dynamic_interruptions: Optional[List["DynamicInterruption"]] = None,
    **kwargs,
) -> Tuple[
    Trajectory[torch.Tensor],
    Tuple["Interruption", ...],
    torch.Tensor,
    State[torch.Tensor],
]:
    nodyn = dynamic_interruptions is None or len(dynamic_interruptions) == 0
    nostat = next_static_interruption is None

    if nostat and nodyn:
        trajectory = simulate(dynamics, start_state, timespan, **kwargs)
        # TODO support event_dim > 0
        return trajectory, (), timespan[-1], trajectory[..., -1]

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
    event_time, event_states = _batched_odeint(  # torchdiffeq.odeint_event(
        functools.partial(_deriv, dynamics, start_state.var_order),
        tuple(getattr(start_state, v) for v in start_state.var_order),
        timespan[0],
        event_fn=combined_event_f,
    )

    # event_state has both the first and final state of the interrupted simulation. We just want the last.
    event_state: Tuple[torch.Tensor, ...] = tuple(
        s[..., -1] for s in event_states
    )  # TODO support event_dim > 0

    # Check which event(s) fired, and put the triggered events in a list.
    # TODO support batched outputs of event functions
    fired_mask = torch.isclose(
        combined_event_f(event_time, event_state),
        torch.tensor(0.0),
        rtol=1e-02,
        atol=1e-03,
    ).reshape(-1)

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
    timespan_2nd_pass = torch.cat(
        (timespan[timespan < event_time], event_time.unsqueeze(0))
    )

    # Execute a standard, non-event based simulation on the new timespan.
    trajectory = simulate(dynamics, start_state, timespan_2nd_pass, **kwargs)

    # Return that trajectory (with interruption time separated out into the end state), the list of triggered
    #  events, the time of the triggered event, and the state at the time of the triggered event.
    # TODO support event_dim > 0
    return (
        trajectory[..., :-1],
        tuple(triggered_events),
        event_time,
        trajectory[..., -1],
    )


# TODO AZ — maybe to multiple dispatch on the interruption type and state type?
def torchdiffeq_point_interruption_flattened_event_f(
    pi: "PointInterruption",
) -> Callable[[torch.Tensor, Tuple[torch.Tensor, ...]], torch.Tensor]:
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
) -> Callable[[torch.Tensor, Tuple[torch.Tensor, ...]], torch.Tensor]:
    """
    Construct a flattened event function for a dynamic interruption.
    :param di: The dynamic interruption for which to build the event function.
    :return: The constructed event function.
    """

    def event_f(t: torch.Tensor, flat_state: Tuple[torch.Tensor, ...]):
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
) -> Callable[[torch.Tensor, Tuple[torch.Tensor, ...]], torch.Tensor]:
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

    def combined_event_f(t: torch.Tensor, flat_state: Tuple[torch.Tensor, ...]):
        return torch.stack(
            list(
                torch.broadcast_tensors(
                    *[f(t, flat_state) for f in dynamic_event_fs],
                    terminal_event_f(t, flat_state),
                )
            ),
            dim=-1,
        )  # TODO support event_dim > 0

    return combined_event_f
