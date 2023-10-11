import functools
import numbers
from typing import (
    TYPE_CHECKING,
    FrozenSet,
    Generic,
    Optional,
    Protocol,
    TypeVar,
    Union,
    runtime_checkable,
)

import pyro
import torch

if TYPE_CHECKING:
    from chirho.dynamical.internals.backend import Solver

R = Union[numbers.Real, torch.Tensor]
S = TypeVar("S")
T = TypeVar("T")
T_co = TypeVar("T_co", covariant=True)


class State(Generic[T]):
    def __init__(self, **values: T):
        # self.class_name =
        self.__dict__["_values"] = {}
        for k, v in values.items():
            setattr(self, k, v)

    @property
    def var_order(self):
        return tuple(sorted(self.keys))

    @property
    def keys(self) -> FrozenSet[str]:
        return frozenset(self.__dict__["_values"].keys())

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.__dict__['_values']})"

    def __str__(self) -> str:
        return f"{self.__class__.__name__}({self.__dict__['_values']})"

    def __setattr__(self, __name: str, __value: T) -> None:
        self.__dict__["_values"][__name] = __value

    def __getattr__(self, __name: str) -> T:
        if __name in self.__dict__["_values"]:
            return self.__dict__["_values"][__name]
        else:
            raise AttributeError(f"{__name} not in {self.__dict__['_values']}")


class _Sliceable(Protocol[T_co]):
    def __getitem__(self, key) -> Union[T_co, "_Sliceable[T_co]"]:
        ...


class Trajectory(Generic[T], State[_Sliceable[T]]):
    def __len__(self) -> int:
        # TODO this implementation is just for tensors, but we should support other types.
        return getattr(self, next(iter(self.keys))).shape[-1]

    def _getitem(self, key):
        from chirho.dynamical.internals._patterns import _index_last_dim_with_mask

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


@runtime_checkable
class InPlaceDynamics(Protocol[S]):
    def diff(self, __state: State[S], __dstate: State[S]) -> None:
        ...


@runtime_checkable
class ObservableInPlaceDynamics(InPlaceDynamics[S], Protocol[S]):
    def diff(self, __state: State[S], __dstate: State[S]) -> None:
        ...

    def observation(self, __state: Union[State[S], Trajectory[S]]) -> None:
        ...


@pyro.poutine.runtime.effectful(type="simulate")
def simulate(
    dynamics: InPlaceDynamics[T],
    initial_state: State[T],
    start_time: R,
    end_time: R,
    *,
    solver: Optional[
        "Solver"
    ] = None,  # Quoted type necessary w/ TYPE_CHECKING to avoid circular import error
    **kwargs,
) -> State[T]:
    """
    Simulate a dynamical system.
    """
    from chirho.dynamical.internals.backend import get_solver, simulate_point

    solver = solver if solver is not None else get_solver()
    return simulate_point(
        solver, dynamics, initial_state, start_time, end_time, **kwargs
    )
