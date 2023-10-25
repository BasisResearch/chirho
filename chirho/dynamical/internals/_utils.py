import functools
import typing
from typing import Any, Callable, Dict, FrozenSet, Optional, Tuple, TypeVar

import pyro
import torch

from chirho.dynamical.ops import State
from chirho.indexed.ops import IndexSet, gather, indices_of, union
from chirho.interventional.handlers import intervene
from chirho.observational.ops import Observation, observe

S = TypeVar("S")
T = TypeVar("T")


@indices_of.register(State)
def _indices_of_state(state: State, *, event_dim: int = 0, **kwargs) -> IndexSet:
    return union(
        *(indices_of(state[k], event_dim=event_dim, **kwargs) for k in state.keys())
    )


@gather.register(State)
def _gather_state(
    state: State[T], indices: IndexSet, *, event_dim: int = 0, **kwargs
) -> State[T]:
    return type(state)(
        **{
            k: gather(state[k], indices, event_dim=event_dim, **kwargs)
            for k in state.keys()
        }
    )


@intervene.register(State)
def _state_intervene(obs: State[T], act: State[T], **kwargs) -> State[T]:
    new_state: State[T] = State()
    for k in obs.keys():
        new_state[k] = intervene(obs[k], act[k] if k in act else None, **kwargs)
    return new_state


@functools.singledispatch
def append(fst, rest: T) -> T:
    raise NotImplementedError(f"append not implemented for type {type(fst)}.")


@append.register(State)
def _append_trajectory(traj1: State[T], traj2: State[T]) -> State[T]:
    if len(traj1.keys()) == 0:
        return traj2

    if len(traj2.keys()) == 0:
        return traj1

    if traj1.keys() != traj2.keys():
        raise ValueError(
            f"Trajectories must have the same keys to be appended, but got {traj1.keys()} and {traj2.keys()}."
        )

    result: State[T] = State()
    for k in traj1.keys():
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
    return State(**{k: traj[k].squeeze(-1) for k in traj.keys()})


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
        **{k: observe(rv[k], obs[k], name=f"{name}__{k}", **kwargs) for k in rv.keys()}
    )


class ShallowMessenger(pyro.poutine.messenger.Messenger):
    """
    Base class for so-called "shallow" effect handlers that uninstall themselves
    after handling a single operation.

    .. warning::

        Does not support post-processing or overriding generic ``_process_message``
    """

    used: bool

    def __enter__(self):
        self.used = False
        return super().__enter__()

    def __exit__(self, exc_type, exc_value, traceback):
        if self in pyro.poutine.runtime._PYRO_STACK:
            super().__exit__(exc_type, exc_value, traceback)

    @typing.final
    def _process_message(self, msg: Dict[str, Any]) -> None:
        if not self.used and hasattr(self, f"_pyro_{msg['type']}"):
            self.used = True
            super()._process_message(msg)

            prev_cont: Optional[Callable[[Dict[str, Any]], None]] = msg["continuation"]

            def cont(msg: Dict[str, Any]) -> None:
                ix = pyro.poutine.runtime._PYRO_STACK.index(self)
                pyro.poutine.runtime._PYRO_STACK.pop(ix)
                if prev_cont is not None:
                    prev_cont(msg)

            msg["continuation"] = cont

    @typing.final
    def _postprocess_message(self, msg: Dict[str, Any]) -> None:
        if hasattr(self, f"_pyro_post_{msg['type']}"):
            raise NotImplementedError("ShallowHandler does not support postprocessing")
