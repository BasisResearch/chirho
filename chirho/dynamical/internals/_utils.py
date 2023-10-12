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


# @indices_of.register(Trajectory)
def _indices_of_trajectory(
    trj: Trajectory, *, event_dim: int = 0, **kwargs
) -> IndexSet:
    return union(
        *(
            indices_of(getattr(trj, k), event_dim=event_dim + 1, **kwargs)
            for k in trj.keys
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


# @gather.register(Trajectory)
def _gather_trajectory(
    trj: Trajectory[T], indices: IndexSet, *, event_dim: int = 0, **kwargs
) -> Trajectory[T]:
    return type(trj)(
        **{
            k: gather(getattr(trj, k), indices, event_dim=event_dim + 1, **kwargs)
            for k in trj.keys
        }
    )


def _index_last_dim_with_mask(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    # Index into the last dimension of x with a boolean mask.
    # TODO AZ — There must be an easier way to do this?
    # NOTE AZ — this could be easily modified to support the last n dimensions, adapt if needed.

    if mask.dtype != torch.bool:
        raise ValueError(
            f"_index_last_dim_with_mask only supports boolean mask indexing, but got dtype {mask.dtype}."
        )

    # Require that the mask is 1d and aligns with the last dimension of x.
    if mask.ndim != 1 or mask.shape[0] != x.shape[-1]:
        raise ValueError(
            "_index_last_dim_with_mask only supports 1d boolean mask indexing, and must align with the last "
            f"dimension of x, but got mask shape {mask.shape} and x shape {x.shape}."
        )

    return torch.masked_select(
        x,
        # Get a shape that will broadcast to the shape of x. This will be [1, ..., len(mask)].
        mask.reshape((1,) * (x.ndim - 1) + mask.shape)
        # masked_select flattens tensors, so we need to reshape back to the original shape w/ the mask applied.
    ).reshape(x.shape[:-1] + (int(mask.sum()),))


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
