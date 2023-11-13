import functools
from typing import Callable, List, Optional, Tuple, TypeVar

import pyro
import torch
import torchdiffeq

from chirho.dynamical.handlers.solver import TorchDiffEq
from chirho.dynamical.internals._utils import _squeeze_time_dim, _var_order
from chirho.dynamical.internals.solver import (
    Interruption,
    get_solver,
    simulate_point,
    simulate_trajectory,
)
from chirho.dynamical.ops import Dynamics, State
from chirho.indexed.ops import IndexSet, gather, get_index_plates

S = TypeVar("S")
T = TypeVar("T")


def _deriv(
    dynamics: Dynamics[torch.Tensor],
    var_order: Tuple[str, ...],
    time: torch.Tensor,
    state: Tuple[torch.Tensor, ...],
) -> Tuple[torch.Tensor, ...]:
    env: State[torch.Tensor] = State()
    for var, value in zip(var_order, state):
        env[var] = value

    assert "t" not in set(env.keys()), "variable name t is reserved for time"
    env["t"] = time

    ddt: State[torch.Tensor] = dynamics(env)
    return tuple(ddt.get(var, torch.tensor(0.0)) for var in var_order)


def _torchdiffeq_ode_simulate_inner(
    dynamics: Dynamics[torch.Tensor],
    initial_state: State[torch.Tensor],
    timespan,
    **odeint_kwargs,
):
    var_order = _var_order(frozenset(initial_state.keys()))  # arbitrary, but fixed

    diff = timespan[:-1] < timespan[1:]

    # We should only encounter collisions at the beginning or end of the timespan.
    if not torch.all(diff[1:-1]):
        raise ValueError(
            "elements of timespan must be strictly increasing, except at endpoints where interruptions can occur."
        )

    # Add a leading "true" to diff for masking, as we've excluded the first element.
    timespan_ = timespan[torch.cat((torch.tensor([True]), diff))]

    # time_dim is set to -1 by convention.
    # TODO: change this when time dim is allowed to vary.
    time_dim = -1

    if torch.any(diff):
        solns = _batched_odeint(  # torchdiffeq.odeint(
            functools.partial(_deriv, dynamics, var_order),
            tuple(initial_state[v] for v in var_order),
            timespan_,
            **odeint_kwargs,
        )
    else:
        solns = tuple(initial_state[v].unsqueeze(time_dim) for v in var_order)

    # As we've already asserted that collisions only happen at the beginning or end of the timespan, we can just
    #  concatenate the initial and final states to get the full trajectory if there are collisions.
    if not diff[0].item():
        solns = tuple(
            torch.cat((s[..., 0].unsqueeze(time_dim), s), dim=time_dim) for s in solns
        )
    if not diff[-1].item() and len(diff) > 1:
        solns = tuple(
            torch.cat((s, s[..., -1].unsqueeze(time_dim)), dim=time_dim) for s in solns
        )

    trajectory: State[torch.Tensor] = State()
    for var, soln in zip(var_order, solns):
        trajectory[var] = soln

    return trajectory


def _batched_odeint(
    func: Callable[[torch.Tensor, Tuple[torch.Tensor, ...]], Tuple[torch.Tensor, ...]],
    y0: Tuple[torch.Tensor, ...],
    t: torch.Tensor,
    *,
    event_fn=None,
    **odeint_kwargs,
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
            func, y0_expanded, t, event_fn=event_fn, **odeint_kwargs
        )
    else:
        yt_raw = torchdiffeq.odeint(func, y0_expanded, t, **odeint_kwargs)

    yt = tuple(
        torch.transpose(
            yt_[(..., None) + yt_.shape[len(yt_.shape) - event_dim :]],
            -len(yt_.shape) - 1,
            -1 - event_dim,
        )[0]
        for yt_ in yt_raw
    )
    return yt if event_fn is None else (event_t, yt)


@simulate_point.register(TorchDiffEq)
@pyro.poutine.runtime.effectful(type="simulate_point")
def torchdiffeq_simulate(
    solver: TorchDiffEq,
    dynamics: Dynamics[torch.Tensor],
    initial_state: State[torch.Tensor],
    start_time: torch.Tensor,
    end_time: torch.Tensor,
) -> State[torch.Tensor]:
    timespan = torch.stack((start_time, end_time))
    trajectory = _torchdiffeq_ode_simulate_inner(
        dynamics, initial_state, timespan, **solver.odeint_kwargs
    )

    # TODO support dim != -1
    idx_name = "__time"
    name_to_dim = {k: f.dim - 1 for k, f in get_index_plates().items()}
    name_to_dim[idx_name] = -1

    final_idx = IndexSet(**{idx_name: {len(timespan) - 1}})
    final_state_traj = gather(trajectory, final_idx, name_to_dim=name_to_dim)
    final_state = _squeeze_time_dim(final_state_traj)
    return final_state


@simulate_trajectory.register(TorchDiffEq)
@pyro.poutine.runtime.effectful(type="simulate_trajectory")
def torchdiffeq_simulate_trajectory(
    solver: TorchDiffEq,
    dynamics: Dynamics[torch.Tensor],
    initial_state: State[torch.Tensor],
    timespan: torch.Tensor,
) -> State[torch.Tensor]:
    return _torchdiffeq_ode_simulate_inner(
        dynamics, initial_state, timespan, **solver.odeint_kwargs
    )


def _torchdiffeq_get_next_interruptions(
    solver: TorchDiffEq,
    dynamics: Dynamics[torch.Tensor],
    start_state: State[torch.Tensor],
    start_time: torch.Tensor,
    interruptions: List[Interruption],
    **kwargs,
) -> Tuple[Tuple[Interruption, ...], torch.Tensor]:
    var_order = _var_order(frozenset(start_state.keys()))  # arbitrary, but fixed

    # Create the event function combining all dynamic events and the terminal (next) static interruption.
    combined_event_f = torchdiffeq_combined_event_f(interruptions, var_order)

    # Simulate to the event execution.
    event_time, event_solutions = _batched_odeint(  # torchdiffeq.odeint_event(
        functools.partial(_deriv, dynamics, var_order),
        tuple(start_state[v] for v in var_order),
        start_time,
        event_fn=combined_event_f,
        **solver.odeint_kwargs,
    )

    # event_state has both the first and final state of the interrupted simulation. We just want the last.
    event_solution: Tuple[torch.Tensor, ...] = tuple(
        s[..., -1] for s in event_solutions
    )  # TODO support event_dim > 0

    # Check which event(s) fired, and put the triggered events in a list.
    # TODO support batched outputs of event functions
    fired_mask = torch.isclose(
        combined_event_f(event_time, event_solution),
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

    if len(fired_mask) != len(interruptions):
        raise RuntimeError(
            "The event function returned an unexpected number of events."
        )

    triggered_events = [de for de, fm in zip(interruptions, fired_mask) if fm]

    return (
        tuple(triggered_events),
        event_time,
    )


def torchdiffeq_simulate_to_interruption(
    interruptions: List[Interruption],
    dynamics: Dynamics[torch.Tensor],
    initial_state: State[torch.Tensor],
    start_time: torch.Tensor,
    end_time: torch.Tensor,
    **kwargs,
) -> Tuple[State[torch.Tensor], torch.Tensor, Optional[Interruption]]:

    if not interruptions:
        from chirho.dynamical.handlers.interruption import StaticInterruption
        interruptions.append(StaticInterruption(end_time))

    solver = get_solver()
    (next_interruption,), interruption_time = _torchdiffeq_get_next_interruptions(
        solver, dynamics, initial_state, start_time, interruptions
    )

    value = simulate_point(solver, dynamics, initial_state, start_time, interruption_time)
    return value, interruption_time, next_interruption


def torchdiffeq_combined_event_f(
    interruptions: List[Interruption],
    var_order: Tuple[str, ...],
) -> Callable[[torch.Tensor, Tuple[torch.Tensor, ...]], torch.Tensor]:
    """
    Construct a combined event function from a list of dynamic interruptions

    :param interruptions: The dynamic interruptions.
    :return: The combined event function, taking in state and time, and returning a vector of floats. When any element
     of this vector is zero, the corresponding event terminates the simulation.
    """

    def combined_event_f(t: torch.Tensor, flat_state: Tuple[torch.Tensor, ...]):
        state: State[torch.Tensor] = State(
            **{k: v for k, v in zip(var_order, flat_state)}
        )

        return torch.stack(
            torch.broadcast_tensors(*[di.event_f(t, state) for di in interruptions]),
            dim=-1,
        )

    return combined_event_f
