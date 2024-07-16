from __future__ import annotations

import functools
from typing import Mapping, Optional, TypeVar

import pyro
import pyro.distributions
import torch
from typing_extensions import ParamSpec

from chirho.indexed.handlers import add_indices
from chirho.indexed.ops import IndexSet, get_index_plates, indices_of
from chirho.observational.ops import AtomicObservation, observe

P = ParamSpec("P")
K = TypeVar("K")
T = TypeVar("T")


@observe.register(int)
@observe.register(float)
@observe.register(bool)
@observe.register(torch.Tensor)
def _observe_deterministic(rv: T, obs: Optional[AtomicObservation[T]] = None, **kwargs):
    """
    Observe a tensor in a probabilistic program.
    """
    rv_dist = pyro.distributions.Delta(
        torch.as_tensor(rv), event_dim=kwargs.pop("event_dim", 0)
    )
    return observe(rv_dist, obs, **kwargs)


@observe.register(pyro.distributions.Distribution)
@pyro.poutine.runtime.effectful(type="observe")
def _observe_distribution(
    rv: pyro.distributions.Distribution,
    obs: Optional[AtomicObservation[T]] = None,
    *,
    name: Optional[str] = None,
    **kwargs,
) -> T:
    if name is None:
        raise ValueError("name must be specified when observing a distribution")

    if callable(obs):
        raise NotImplementedError("Dependent observations are not yet supported")

    return pyro.sample(name, rv, obs=obs, **kwargs)


@observe.register(dict)
def _observe_dict(
    rv: Mapping[K, T],
    obs: Optional[AtomicObservation[Mapping[K, T]]] = None,
    *,
    name: Optional[str] = None,
    **kwargs,
) -> Mapping[K, T]:
    if callable(obs):
        obs = obs(rv)
        if obs is not rv and obs is not None:
            raise NotImplementedError("Dependent observations are not yet supported")

    if obs is rv or obs is None:
        return rv

    return {k: observe(rv[k], obs[k], name=f"{name}__{k}", **kwargs) for k in rv.keys()}


class ObserveNameMessenger(pyro.poutine.messenger.Messenger):
    def _pyro_observe(self, msg):
        if "name" not in msg["kwargs"]:
            msg["kwargs"]["name"] = msg["name"]


def site_is_delta(msg: dict) -> bool:
    d = msg["fn"]
    while hasattr(d, "base_dist"):
        d = d.base_dist
    return isinstance(d, pyro.distributions.Delta)


@functools.singledispatch
def unbind_leftmost_dim(v, name: str, size: int = 1, **kwargs):
    """
    Helper function to move the leftmost dimension of a ``torch.Tensor``
    or ``pyro.distributions.Distribution`` or other batched value
    into a fresh named dimension using the machinery in ``chirho.indexed`` ,
    allocating a new dimension with the given name if necessary
    via an enclosing :class:`~chirho.indexed.handlers.IndexPlatesMessenger` .

    .. warning:: Must be used in conjunction with :class:`~chirho.indexed.handlers.IndexPlatesMessenger` .

    :param v: Batched value.
    :param name: Name of the fresh dimension.
    :param size: Size of the fresh dimension. If 1, the size is inferred from ``v`` .
    """
    raise NotImplementedError


@unbind_leftmost_dim.register
def _unbind_leftmost_dim_tensor(
    v: torch.Tensor, name: str, size: int = 1, *, event_dim: int = 0
) -> torch.Tensor:
    size = max(size, v.shape[0])
    v = v.expand((size,) + v.shape[1:])

    if name not in get_index_plates():
        add_indices(IndexSet(**{name: set(range(size))}))

    new_dim: int = get_index_plates()[name].dim
    orig_shape = v.shape
    while new_dim - event_dim < -len(v.shape):
        v = v[None]
    if v.shape[0] == 1 and orig_shape[0] != 1:
        v = torch.transpose(v, -len(orig_shape), new_dim - event_dim)
    return v


@unbind_leftmost_dim.register
def _unbind_leftmost_dim_distribution(
    v: pyro.distributions.Distribution, name: str, size: int = 1, **kwargs
) -> pyro.distributions.Distribution:
    size = max(size, v.batch_shape[0])
    if v.batch_shape[0] != 1:
        raise NotImplementedError("Cannot freely reshape distribution")

    if name not in get_index_plates():
        add_indices(IndexSet(**{name: set(range(size))}))

    new_dim: int = get_index_plates()[name].dim
    orig_shape = v.batch_shape

    new_shape = (size,) + (1,) * (-new_dim - len(orig_shape)) + orig_shape[1:]
    return v.expand(new_shape)


@functools.singledispatch
def bind_leftmost_dim(v, name: str, **kwargs):
    """
    Helper function to move a named dimension managed by ``chirho.indexed``
    into a new unnamed dimension to the left of all named dimensions in the value.

    .. warning:: Must be used in conjunction with :class:`~chirho.indexed.handlers.IndexPlatesMessenger` .
    """
    raise NotImplementedError


@bind_leftmost_dim.register
def _bind_leftmost_dim_tensor(
    v: torch.Tensor, name: str, *, event_dim: int = 0, **kwargs
) -> torch.Tensor:
    if name not in indices_of(v, event_dim=event_dim):
        return v
    return torch.transpose(
        v[None], -len(v.shape) - 1, get_index_plates()[name].dim - event_dim
    )
