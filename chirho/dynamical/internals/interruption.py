from typing import TYPE_CHECKING, List, Optional, Tuple, TypeVar

if TYPE_CHECKING:
    from chirho.dynamical.handlers import (
        DynamicInterruption,
        PointInterruption,
        Interruption,
    )

import functools

import pyro
import torch

from chirho.dynamical.handlers.solver import Solver
from chirho.dynamical.ops.dynamical import Dynamics, State, Trajectory, simulate

S = TypeVar("S")
T = TypeVar("T")


# Separating out the effectful operation from the non-effectful dispatch on the default implementation
@pyro.poutine.runtime.effectful(type="simulate_to_interruption")
@pyro.poutine.block(hide_types=["simulate"])  # TODO: Unblock this simulate call
def simulate_to_interruption(
    dynamics: Dynamics[S, T],
    start_state: State[T],
    start_time: T,
    end_time: T,
    *,
    solver: Optional[Solver] = None,
    next_static_interruption: Optional["PointInterruption"] = None,
    dynamic_interruptions: Optional[List["DynamicInterruption"]] = None,
    **kwargs,
) -> Tuple[State[T], Tuple["Interruption", ...], T]:
    """
    Simulate a dynamical system until the next interruption. This will be either one of the passed
    dynamic interruptions, the next static interruption, or the end time, whichever comes first.
    :returns: the final state, a collection of interruptions that ended the simulation
    (this will usually just be a single interruption), and the time the interruption occurred.
    """
    nodyn = dynamic_interruptions is None or len(dynamic_interruptions) == 0
    nostat = next_static_interruption is None

    if nostat and nodyn:
        end_state = simulate(dynamics, start_state, start_time, end_time, solver=solver)
        return end_state, (), end_time

    if nodyn:
        next_static_interruption: PointInterruption  # Linter needs some help here.

        event_state = simulate(
            dynamics,
            start_state,
            start_time,
            next_static_interruption.time,
            solver=solver,
        )
        return event_state, (next_static_interruption,), next_static_interruption.time

    if dynamic_interruptions is None:
        dynamic_interruptions = []

    if next_static_interruption is None:
        next_static_interruption = PointInterruption(time=end_time)

    interruption_time, interruptions = get_next_interruptions(
        dynamics,
        start_state,
        start_time,
        solver=solver,
        next_static_interruption=next_static_interruption,
        dynamic_interruptions=dynamic_interruptions,
        **kwargs,
    )
    event_state = simulate(
        dynamics, start_state, start_time, interruption_time, solver=solver
    )
    return event_state, interruptions, interruption_time


# noinspection PyUnusedLocal
@functools.singledispatch
def get_next_interruptions(
    dynamics: Dynamics[S, T],
    start_state: State[T],
    start_time: T,
    *,
    solver: Optional[Solver] = None,
    next_static_interruption: Optional["PointInterruption"] = None,
    dynamic_interruptions: Optional[List["DynamicInterruption"]] = None,
    **kwargs,
) -> Tuple[T, Tuple["Interruption", ...]]:
    raise NotImplementedError(
        f"get_next_interruptions not implemented for type {type(dynamics)}"
    )


@pyro.poutine.runtime.effectful(type="apply_interruptions")
def apply_interruptions(
    dynamics: Dynamics[S, T], start_state: State[T]
) -> Tuple[Dynamics[S, T], State[T]]:
    """
    Apply the effects of an interruption to a dynamical system.
    """
    # Default is to do nothing.
    return dynamics, start_state


@functools.singledispatch
def concatenate(*inputs, **kwargs):
    """
    Concatenate multiple inputs of type T into a single output of type T.
    """
    raise NotImplementedError(f"concatenate not implemented for type {type(inputs[0])}")


@concatenate.register(Trajectory)
def trajectory_concatenate(*trajectories: Trajectory[T], **kwargs) -> Trajectory[T]:
    """
    Concatenate multiple trajectories into a single trajectory.
    """
    full_trajectory: Trajectory[T] = Trajectory()
    for trajectory in trajectories:
        for k in trajectory.keys:
            if k not in full_trajectory.keys:
                setattr(full_trajectory, k, getattr(trajectory, k))
            else:
                prev_v = getattr(full_trajectory, k)
                curr_v = getattr(trajectory, k)
                time_dim = -1  # TODO generalize to nontrivial event_shape
                batch_shape = torch.broadcast_shapes(
                    prev_v.shape[:-1], curr_v.shape[:-1]
                )
                prev_v = prev_v.expand(*batch_shape, *prev_v.shape[-1:])
                curr_v = curr_v.expand(*batch_shape, *curr_v.shape[-1:])
                setattr(
                    full_trajectory,
                    k,
                    torch.cat([prev_v, curr_v], dim=time_dim),
                )
    return full_trajectory
