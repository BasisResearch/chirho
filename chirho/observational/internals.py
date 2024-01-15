from __future__ import annotations

from typing import Mapping, Optional, TypeVar

import pyro
import pyro.distributions
import torch

from chirho.observational.ops import AtomicObservation, observe

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
