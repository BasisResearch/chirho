import functools
from typing import TypeVar

import torch

from chirho.dynamical.ops.dynamical import State, Trajectory
from chirho.indexed.ops import IndexSet, gather, indices_of, union
from chirho.interventional.handlers import intervene

S = TypeVar("S")
T = TypeVar("T")


@indices_of.register(State)
def _indices_of_state(state: State, *, event_dim: int = 0, **kwargs) -> IndexSet:
    return union(
        *(
            indices_of(getattr(state, k), event_dim=event_dim, **kwargs)
            for k in state.keys
        )
    )


@gather.register(State)
def _gather_state(
    state: State[T], indices: IndexSet, *, event_dim: int = 0, **kwargs
) -> State[T]:
    return type(state)(
        **{
            k: gather(getattr(state, k), indices, event_dim=event_dim, **kwargs)
            for k in state.keys
        }
    )


@intervene.register(State)
def _state_intervene(obs: State[T], act: State[T], **kwargs) -> State[T]:
    new_state: State[T] = State()
    for k in obs.keys:
        setattr(
            new_state, k, intervene(getattr(obs, k), getattr(act, k, None), **kwargs)
        )
    return new_state


@functools.singledispatch
def append(fst, rest: T) -> T:
    raise NotImplementedError(f"append not implemented for type {type(fst)}.")


@append.register(Trajectory)
def _append_trajectory(traj1: Trajectory[T], traj2: Trajectory[T]) -> Trajectory[T]:
    if len(traj1.keys) == 0:
        return traj2

    if len(traj2.keys) == 0:
        return traj1

    if traj1.keys != traj2.keys:
        raise ValueError(
            f"Trajectories must have the same keys to be appended, but got {traj1.keys} and {traj2.keys}."
        )

    result: Trajectory[T] = Trajectory()
    for k in traj1.keys:
        setattr(result, k, append(getattr(traj1, k), getattr(traj2, k)))

    return result


@append.register(torch.Tensor)
def _append_tensor(prev_v: torch.Tensor, curr_v: torch.Tensor) -> torch.Tensor:
    time_dim = -1  # TODO generalize to nontrivial event_shape
    batch_shape = torch.broadcast_shapes(prev_v.shape[:-1], curr_v.shape[:-1])
    prev_v = prev_v.expand(*batch_shape, *prev_v.shape[-1:])
    curr_v = curr_v.expand(*batch_shape, *curr_v.shape[-1:])
    return torch.cat([prev_v, curr_v], dim=time_dim)
