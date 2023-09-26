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
from chirho.dynamical.ops.dynamical import Dynamics, State, Trajectory

S = TypeVar("S")
T = TypeVar("T")


# Separating out the effectful operation from the non-effectful dispatch on the default implementation
@pyro.poutine.runtime.effectful(type="simulate_to_interruption")
@pyro.poutine.block(hide_types=["simulate"])
def simulate_to_interruption(
    dynamics: Dynamics[S, T],
    start_state: State[T],
    timespan,  # The first element of timespan is assumed to be the starting time.
    *,
    solver: Optional[Solver] = None,
    next_static_interruption: Optional["PointInterruption"] = None,
    dynamic_interruptions: Optional[List["DynamicInterruption"]] = None,
    **kwargs,
) -> Tuple[Trajectory[T], Tuple["Interruption", ...], T, State[T]]:
    """
    Simulate a dynamical system until the next interruption. Return the state at the requested time points, and
     a collection of interruptions that ended the simulation (this will usually just be a single interruption).
    This will be either one of the passed dynamic interruptions or the next static interruption, whichever comes
     first.
    :returns: the state at the requested time points, the interruption that ended the simulation, the time at which
     the simulation ended, and the end state. The initial trajectory object does not include state measurements at
     the end-point.
    """
    return _simulate_to_interruption(
        dynamics,
        start_state,
        timespan,
        solver=solver,
        next_static_interruption=next_static_interruption,
        dynamic_interruptions=dynamic_interruptions,
        **kwargs,
    )


# noinspection PyUnusedLocal
@functools.singledispatch
def _simulate_to_interruption(
    dynamics: Dynamics[S, T],
    start_state: State[T],
    timespan,  # The first element of timespan is assumed to be the starting time.
    *,
    solver: Optional[Solver] = None,
    next_static_interruption: Optional["PointInterruption"] = None,
    dynamic_interruptions: Optional[List["DynamicInterruption"]] = None,
    **kwargs,
) -> Tuple[Trajectory[T], Tuple["Interruption", ...], T, State[T]]:
    raise NotImplementedError(
        f"simulate_to_interruption not implemented for type {type(dynamics)}"
    )


simulate_to_interruption.register = _simulate_to_interruption.register


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
