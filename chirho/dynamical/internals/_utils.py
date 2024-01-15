import collections.abc
import dataclasses
import functools
import typing
from typing import Any, Callable, Dict, FrozenSet, Generic, Optional, Tuple, TypeVar

import pyro
import torch

from chirho.dynamical.ops import State

S = TypeVar("S")
T = TypeVar("T")


@functools.singledispatch
def append(fst, rest: T) -> T:
    raise NotImplementedError(f"append not implemented for type {type(fst)}.")


@append.register(collections.abc.Mapping)
def _append_trajectory(traj1: State[T], traj2: State[T]) -> State[T]:
    if len(traj1.keys()) == 0:
        return traj2

    if len(traj2.keys()) == 0:
        return traj1

    if traj1.keys() != traj2.keys():
        raise ValueError(
            f"Trajectories must have the same keys to be appended, but got {traj1.keys()} and {traj2.keys()}."
        )

    return type(traj1)(**{k: append(traj1[k], traj2[k]) for k in traj1.keys()})


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


@functools.singledispatch
def _squeeze_time_dim(state_or_traj):
    raise NotImplementedError(
        f"_squeeze_time_dim not implemented for type {type(state_or_traj)}."
    )


@_squeeze_time_dim.register(collections.abc.Mapping)
def _squeeze_time_dim_trajectory(traj: State[T]) -> State[T]:
    return type(traj)(**{k: _squeeze_time_dim(traj[k]) for k in traj.keys()})


@_squeeze_time_dim.register(torch.Tensor)
def _squeeze_time_dim_tensor(state: torch.Tensor) -> torch.Tensor:
    return state.squeeze(-1)


@functools.singledispatch
def _unsqueeze_time_dim(state_or_traj):
    raise NotImplementedError(
        f"_unsqueeze_time_dim not implemented for type {type(state_or_traj)}."
    )


@_unsqueeze_time_dim.register(collections.abc.Mapping)
def _unsqueeze_time_dim_state(state: State[T]) -> State[T]:
    return type(state)(**{k: _unsqueeze_time_dim(state[k]) for k in state.keys()})


@_unsqueeze_time_dim.register(torch.Tensor)
def _unsqueeze_time_dim_tensor(state: torch.Tensor) -> torch.Tensor:
    return state.unsqueeze(-1)


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


@dataclasses.dataclass(order=True)
class Prioritized(Generic[T]):
    priority: float
    item: T = dataclasses.field(compare=False)
