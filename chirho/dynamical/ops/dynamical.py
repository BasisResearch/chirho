import functools
from typing import Callable, FrozenSet, Generic, Protocol, TypeVar, runtime_checkable

import pyro
import torch

from chirho.dynamical.internals.dynamical import _index_last_dim_with_mask

S = TypeVar("S")
T = TypeVar("T")


class State(Generic[T]):
    def __init__(self, **values: T):
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
        return f"State({self.__dict__['_values']})"

    def __str__(self) -> str:
        return f"State({self.__dict__['_values']})"

    def __setattr__(self, __name: str, __value: T) -> None:
        self.__dict__["_values"][__name] = __value

    def __getattr__(self, __name: str) -> T:
        if __name in self.__dict__["_values"]:
            return self.__dict__["_values"][__name]
        else:
            raise AttributeError(f"{__name} not in {self.__dict__['_values']}")

    # TODO doesn't allow for explicitly handling mismatched keys.
    # def __sub__(self, other: 'State[T]') -> 'State[T]':
    #     # TODO throw errors if keys don't match, or if shapes don't match...but that should be the job of traj?
    #     return State(**{k: getattr(self, k) - getattr(other, k) for k in self.keys})

    def subtract_shared_variables(self, other: "State[T]"):
        shared_keys = self.keys.intersection(other.keys)
        return State(**{k: getattr(self, k) - getattr(other, k) for k in shared_keys})

    # FIXME AZ - non-generic method in generic class.
    def l2(self) -> torch.Tensor:
        """
        Compute the L2 norm of the state. This is useful e.g. after taking the difference between two states.
        :return: The L2 norm of the vectorized state.
        """
        return torch.sqrt(
            torch.sum(
                torch.square(torch.tensor(*[getattr(self, k) for k in self.keys]))
            )
        )

    def trajectorify(self) -> "Trajectory[T]":
        ret: Trajectory[T] = Trajectory(
            # TODO support event_dim > 0
            **{k: getattr(self, k)[..., None] for k in self.keys}
        )
        return ret


# TODO AZ - this differentiation needs to go away probably...this is useful for us during dev to be clear about when
#  we expect multiple vs. a single state in the vectors, but it's likely confusing/not useful for the user? Maybe,
#  maybe not. If we do keep it we need more explicit guarantees that the State won't have more than a single entry?
class Trajectory(State[T]):
    def __init__(self, **values: T):
        super().__init__(**values)

    def _getitem(self, key):
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


@runtime_checkable
class Dynamics(Protocol[S, T]):
    diff: Callable[[State[S], State[S]], T]


@pyro.poutine.runtime.effectful(type="simulate")
def simulate(
    dynamics: Dynamics[S, T], initial_state: State[T], timespan, **kwargs
) -> Trajectory[T]:
    """
    Simulate a dynamical system.
    """
    return _simulate(dynamics, initial_state, timespan, **kwargs)


# This redirection distinguishes between the effectful operation, and the
# type-directed dispatch on Dynamics
@functools.singledispatch
def _simulate(
    dynamics: Dynamics[S, T], initial_state: State[T], timespan, **kwargs
) -> Trajectory[T]:
    """
    Simulate a dynamical system.
    """
    raise NotImplementedError(f"simulate not implemented for type {type(dynamics)}")


simulate.register = _simulate.register
