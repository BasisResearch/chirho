import functools
from typing import FrozenSet, Optional, Tuple, TypeVar

import torch

from chirho.dynamical.ops import State, get_keys
from chirho.indexed.ops import IndexSet, gather, indices_of, union
from chirho.interventional.handlers import intervene
from chirho.observational.ops import Observation, observe

S = TypeVar("S")
T = TypeVar("T")


@indices_of.register(State)
def _indices_of_state(state: State, *, event_dim: int = 0, **kwargs) -> IndexSet:
    return union(
        *(indices_of(state[k], event_dim=event_dim, **kwargs) for k in get_keys(state))
    )


@gather.register(State)
def _gather_state(
    state: State[T], indices: IndexSet, *, event_dim: int = 0, **kwargs
) -> State[T]:
    return type(state)(
        **{
            k: gather(state[k], indices, event_dim=event_dim, **kwargs)
            for k in get_keys(state)
        }
    )


@intervene.register(State)
def _state_intervene(obs: State[T], act: State[T], **kwargs) -> State[T]:
    new_state: State[T] = State()
    for k in get_keys(obs):
        new_state[k] = intervene(obs[k], act[k] if k in act else None, **kwargs)
    return new_state


@functools.singledispatch
def append(fst, rest: T) -> T:
    raise NotImplementedError(f"append not implemented for type {type(fst)}.")


@append.register(State)
def _append_trajectory(traj1: State[T], traj2: State[T]) -> State[T]:
    if len(get_keys(traj1)) == 0:
        return traj2

    if len(get_keys(traj2)) == 0:
        return traj1

    if get_keys(traj1) != get_keys(traj2):
        raise ValueError(
            f"Trajectories must have the same keys to be appended, but got {get_keys(traj1)} and {get_keys(traj2)}."
        )

    result: State[T] = State()
    for k in get_keys(traj1):
        result[k] = append(traj1[k], traj2[k])
    return result


@append.register(torch.Tensor)
def _append_tensor(prev_v: torch.Tensor, curr_v: torch.Tensor) -> torch.Tensor:
    time_dim = -1  # TODO generalize to nontrivial event_shape
    batch_shape = torch.broadcast_shapes(prev_v.shape[:-1], curr_v.shape[:-1])
    prev_v = prev_v.expand(*batch_shape, *prev_v.shape[-1:])
    curr_v = curr_v.expand(*batch_shape, *curr_v.shape[-1:])
    return torch.cat([prev_v, curr_v], dim=time_dim)


@functools.lru_cache
def _var_order(varnames: FrozenSet[str]) -> Tuple[str, ...]:
    return tuple(sorted(varnames))


def _squeeze_time_dim(traj: State[torch.Tensor]) -> State[torch.Tensor]:
    return State(**{k: traj[k].squeeze(-1) for k in get_keys(traj)})


@observe.register(State)
def _observe_state(
    rv: State[T],
    obs: Optional[Observation[State[T]]] = None,
    *,
    name: Optional[str] = None,
    **kwargs,
) -> State[T]:
    if callable(obs):
        obs = obs(rv)
        if obs is not rv and obs is not None:
            raise NotImplementedError("Dependent observations are not yet supported")

    if obs is rv or obs is None:
        return rv

    assert isinstance(obs, State)

    return State(
        **{
            k: observe(rv[k], obs[k], name=f"{name}__{k}", **kwargs)
            for k in get_keys(rv)
        }
    )
