import functools
import typing
from typing import Generic, Protocol, TypeVar

import torch

from chirho.dynamical.ops.dynamical import Dynamics, State

if typing.TYPE_CHECKING:
    from chirho.dynamical.handlers.solver import Solver

S = TypeVar("S")
T = TypeVar("T")
T_co = TypeVar("T_co", covariant=True)


class _Sliceable(Protocol[T_co]):
    def __getitem__(self, key) -> T_co | "_Sliceable[T_co]":
        ...


class Trajectory(Generic[T], State[_Sliceable[T]]):
    def __len__(self) -> int:
        # TODO this implementation is just for tensors, but we should support other types.
        return getattr(self, next(iter(self.keys))).shape[-1]

    def _getitem(self, key):
        from chirho.dynamical.internals.indexed import _index_last_dim_with_mask

        if isinstance(key, str):
            raise ValueError(
                "Trajectory does not support string indexing, use getattr instead if you want to access a specific "
                "state variable."
            )

        item = State() if isinstance(key, int) else Trajectory()
        for k, v in self.__dict__["_values"].items():
            if isinstance(key, torch.Tensor):
                keyd_v = _index_last_dim_with_mask(v, key)
            else:
                keyd_v = v[key]
            setattr(item, k, keyd_v)
        return item

    # This is needed so that mypy and other type checkers believe that Trajectory can be indexed into.
    @functools.singledispatchmethod
    def __getitem__(self, key):
        return self._getitem(key)

    @__getitem__.register(int)
    def _getitem_int(self, key: int) -> State[T]:
        return self._getitem(key)

    @__getitem__.register(torch.Tensor)
    def _getitem_torchmask(self, key: torch.Tensor) -> "Trajectory[T]":
        if key.dtype != torch.bool:
            raise ValueError(
                f"__getitem__ with a torch.Tensor only supports boolean mask indexing, but got dtype {key.dtype}."
            )

        return self._getitem(key)

    def to_state(self) -> State[T]:
        ret: State[T] = State(
            # TODO support event_dim > 0
            **{k: getattr(self, k) for k in self.keys}
        )
        return ret

    @functools.singledispatchmethod
    def append(self, other: "Trajectory[T]"):
        raise NotImplementedError(f"append not implemented for type {type(other)}")


@Trajectory.append.register(Trajectory)  # type: ignore
def _append_trajectory(self: Trajectory[torch.Tensor], other: Trajectory[torch.Tensor]):
    # If self is empty, just copy other.
    if len(self.keys) == 0:
        for k in other.keys:
            setattr(self, k, getattr(other, k))
        return

    if self.keys != other.keys:
        raise ValueError(
            f"Trajectories must have the same keys to be appended, but got {self.keys} and {other.keys}."
        )
    for k in self.keys:
        prev_v = getattr(self, k)
        curr_v = getattr(other, k)
        time_dim = -1  # TODO generalize to nontrivial event_shape
        batch_shape = torch.broadcast_shapes(prev_v.shape[:-1], curr_v.shape[:-1])
        prev_v = prev_v.expand(*batch_shape, *prev_v.shape[-1:])
        curr_v = curr_v.expand(*batch_shape, *curr_v.shape[-1:])
        setattr(
            self,
            k,
            torch.cat([prev_v, curr_v], dim=time_dim),
        )


@functools.singledispatch
def simulate_trajectory(
    solver: "Solver",  # Quoted type necessary w/ TYPE_CHECKING to avoid circular import error
    dynamics: Dynamics[S, T],
    initial_state: State[T],
    timespan: T,
    **kwargs,
) -> Trajectory[T]:
    """
    Simulate a dynamical system.
    """
    raise NotImplementedError(
        f"simulate_trajectory not implemented for solver of type {type(solver)}"
    )
