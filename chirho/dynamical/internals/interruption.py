import functools
from typing import List, Optional, Tuple, TypeVar

import pyro
import torch

from chirho.dynamical.handlers.interruption import (
    DynamicInterruption,
    Interruption,
    StaticInterruption,
)
from chirho.dynamical.handlers.solver import Solver
from chirho.dynamical.ops.dynamical import Dynamics, State, Trajectory, simulate

S = TypeVar("S")
T = TypeVar("T")


# Separating out the effectful operation from the non-effectful dispatch on the default implementation
@pyro.poutine.runtime.effectful(type="simulate_to_interruption")
# @pyro.poutine.block(hide_types=["simulate"])  # TODO: Unblock this simulate call
def simulate_to_interruption(
    dynamics: Dynamics[S, T],
    start_state: State[T],
    start_time: T,
    end_time: T,
    *,
    solver: Optional[Solver] = None,
    next_static_interruption: Optional["StaticInterruption"] = None,
    dynamic_interruptions: List["DynamicInterruption"] = [],
    **kwargs,
) -> Tuple[State[T], Tuple["Interruption", ...], T]:
    """
    Simulate a dynamical system until the next interruption. This will be either one of the passed
    dynamic interruptions, the next static interruption, or the end time, whichever comes first.
    :returns: the final state, a collection of interruptions that ended the simulation
    (this will usually just be a single interruption), and the time the interruption occurred.
    """
    interruptions, interruption_time = get_next_interruptions(
        dynamics,
        start_state,
        start_time,
        end_time,
        solver=solver,
        next_static_interruption=next_static_interruption,
        dynamic_interruptions=dynamic_interruptions,
        **kwargs,
    )

    event_state = simulate(
        dynamics, start_state, start_time, interruption_time, solver=solver
    )
    return event_state, interruptions, interruption_time


def get_next_interruptions(
    dynamics: Dynamics[S, T],
    start_state: State[T],
    start_time: T,
    end_time: T,
    *,
    solver: Optional[Solver] = None,
    next_static_interruption: Optional["StaticInterruption"] = None,
    dynamic_interruptions: List["DynamicInterruption"] = [],
    **kwargs,
) -> Tuple[Tuple["Interruption", ...], T]:

    # This is necessary to get the linter to understand the type of various variables.
    interruptions: Tuple[Interruption, ...]
    interruption_time: T

    nodyn = len(dynamic_interruptions) == 0
    nostat = next_static_interruption is None

    if nostat or next_static_interruption.time > end_time:  # type: ignore
        # If there's no static interruption or the next static interruption is after the end time,
        # we'll just simulate until the end time.
        next_static_interruption = StaticInterruption(time=end_time)

    assert type(next_static_interruption) is StaticInterruption  # Linter needs a hint

    if nodyn:
        # If there's no dynamic intervention, we'll simulate until either the end_time,
        # or the `next_static_interruption` whichever comes first.
        interruptions = (next_static_interruption,)
        interruption_time = next_static_interruption.time  # type: ignore
    else:
        interruptions, interruption_time = get_next_interruptions_dynamic(  # type: ignore
            dynamics,  # type: ignore
            start_state,  # type: ignore
            start_time,  # type: ignore
            solver=solver,
            next_static_interruption=next_static_interruption,
            dynamic_interruptions=dynamic_interruptions,
            **kwargs,
        )

    return interruptions, interruption_time


# noinspection PyUnusedLocal
@functools.singledispatch
def get_next_interruptions_dynamic(
    dynamics: Dynamics[S, T],
    start_state: State[T],
    start_time: T,
    next_static_interruption: StaticInterruption,
    dynamic_interruptions: List[DynamicInterruption],
    *,
    solver: Optional[Solver] = None,
    **kwargs,
) -> Tuple[Tuple[Interruption, ...], T]:
    raise NotImplementedError(
        f"get_next_interruptions_dynamic not implemented for type {type(dynamics)}"
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
