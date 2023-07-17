from typing import (
    TYPE_CHECKING,
    Callable,
    FrozenSet,
    Generic,
    List,
    Optional,
    Protocol,
    Tuple,
    TypeVar,
    runtime_checkable,
)

if TYPE_CHECKING:
    from .handlers import DynamicInterruption, PointInterruption, Interruption

import functools

import pyro
import torch

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


@functools.singledispatch
def unsqueeze(x, axis: int):
    raise NotImplementedError(f"unsqueeze not implemented for type {type(x)}")


@unsqueeze.register
def _unsqueeze_torch(x: torch.Tensor, axis: int) -> torch.Tensor:
    return torch.unsqueeze(x, axis)


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


@functools.singledispatch
@pyro.poutine.runtime.effectful(type="apply_interruptions")
def apply_interruptions(
    dynamics: Dynamics[S, T], start_state: State[T]
) -> Tuple[Dynamics[S, T], State[T]]:
    """
    Apply the effects of an interruption to a dynamical system.
    """
    # Default is to do nothing.
    return dynamics, start_state


# noinspection PyUnusedLocal
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


# noinspection PyUnusedLocal
@functools.singledispatch
def simulate(
    dynamics: Dynamics[S, T], initial_state: State[T], timespan, **kwargs
) -> Trajectory[T]:
    """
    Simulate a dynamical system.
    """
    raise NotImplementedError(f"simulate not implemented for type {type(dynamics)}")
