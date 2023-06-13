from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Generic,
    Hashable,
    List,
    Mapping,
    Optional,
    Protocol,
    Set,
    Tuple,
    TypeVar,
    Union,
    runtime_checkable,
)

if TYPE_CHECKING:
    from .handlers import DynamicInterruption, PointInterruption, Interruption

import functools

import pyro
import torch

S, T = TypeVar("S"), TypeVar("T")


class State(Generic[T]):
    def __init__(self, **values: T):
        self.__dict__["_values"]: dict[str, T] = {}
        for k, v in values.items():
            setattr(self, k, v)

    @property
    def var_order(self):
        return tuple(sorted(self.keys))

    @property
    def keys(self) -> Set[str]:
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
        return super().__getattr__(__name)

    # TODO doesn't allow for explicitly handling mismatched keys.
    # def __sub__(self, other: 'State[T]') -> 'State[T]':
    #     # TODO throw errors if keys don't match, or if shapes don't match...but that should be the job of traj?
    #     return State(**{k: getattr(self, k) - getattr(other, k) for k in self.keys})

    def subtract_shared_variables(self, other: "State[T]"):
        shared_keys = self.keys.intersection(other.keys)
        return State(**{k: getattr(self, k) - getattr(other, k) for k in shared_keys})

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
        ret = Trajectory(**{k: torch.unsqueeze(getattr(self, k), 0) for k in self.keys})
        return ret


# TODO AZ - this differentiation needs to go away probably...this is useful for us during dev to be clear about when
#  we expect multiple vs. a single state in the vectors, but it's likely confusing/not useful for the user? Maybe,
#  maybe not. If we do keep it we need more explicit guarantees that the State won't have more than a single entry?
class Trajectory(State[T]):
    def __init__(self, **values: T):
        super().__init__(**values)

    def __getitem__(self, key: Union[int, slice]) -> Union[State[T], "Trajectory[T]"]:
        if isinstance(key, str):
            raise ValueError(
                "Trajectory does not support string indexing, use getattr instead if you want to access a specific "
                "state variable."
            )

        state = State() if isinstance(key, int) else Trajectory()
        for k, v in self.__dict__["_values"].items():
            setattr(state, k, v[key])
        return state


@runtime_checkable
class Dynamics(Protocol[S, T]):
    diff: Callable[[State[S], State[S]], T]


@functools.singledispatch
def simulate_span(
    dynamics: Dynamics[S, T],
    curr_state: State[T],
    timespan,
    event_fn: Optional[Callable[[torch.tensor, torch.tensor], torch.tensor]] = None,
    **kwargs,
) -> Trajectory[T]:
    """
    Simulate a fixed timespan of a dynamical system.
    """
    raise NotImplementedError(
        f"simulate_span not implemented for type {type(dynamics)}"
    )


@functools.singledispatch
@pyro.poutine.runtime.effectful(type="apply_interruptions")
def apply_interruptions(
    dynamics: Dynamics[S, T],
    start_state: State[T]
) -> Tuple[Dynamics[S, T], State[T]]:
    """
    Apply the effects of an interruption to a dynamical system.
    """
    # Default is to do nothing.
    return dynamics, start_state


@functools.singledispatch
def simulate_to_interruption(
    dynamics: Dynamics[S, T],
    start_state: State[T],
    timespan,  # The first element of timespan is assumed to be the starting time.
    *,
    next_static_interruption: Optional["PointInterruption"] = None,
    dynamic_interruptions: Optional[List["DynamicInterruption"]] = None,
    **kwargs,
) -> Tuple[Trajectory[T], Tuple["Interruption", ...], float, State[T]]:
    """
    Simulate a dynamical system until the next interruption. Return the state at the requested time points, and
     a collection of interruptions that ended the simulation (this will usually just be a single interruption).
    This will be either one of the passed dynamic interruptions or the next static interruption, whichever comes
     first.
    :returns: the state at the requested time points, the interruption that ended the simulation, the time at which
     the simulation ended, and the end state. The initial trajectory object does not include state measurements at
     the end-point.
    """
    raise NotImplementedError(
        f"simulate_to_interruption not implemented for type {type(dynamics)}"
    )


@functools.singledispatch
def concatenate(*inputs):
    """
    Concatenate multiple inputs of type T into a single output of type T.
    """
    raise NotImplementedError(f"concatenate not implemented for type {type(inputs[0])}")


@concatenate.register
def trajectory_concatenate(*trajectories: Trajectory) -> Trajectory[T]:
    """
    Concatenate multiple trajectories into a single trajectory.
    """
    full_trajectory = Trajectory()
    for trajectory in trajectories:
        for k in trajectory.keys:
            if k not in full_trajectory.keys:
                setattr(full_trajectory, k, getattr(trajectory, k))
            else:
                setattr(
                    full_trajectory,
                    k,
                    torch.cat([getattr(full_trajectory, k), getattr(trajectory, k)]),
                )
    return full_trajectory


@functools.singledispatch
def simulate(
    dynamics: Dynamics[S, T], initial_state: State[T], timespan, **kwargs
) -> Trajectory[T]:
    """
    Simulate a dynamical system.
    """
    raise NotImplementedError(f"simulate not implemented for type {type(dynamics)}")
