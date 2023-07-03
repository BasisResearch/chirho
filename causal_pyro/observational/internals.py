from typing import Optional, TypeVar

import pyro
import torch

from causal_pyro.observational.ops import AtomicObservation, observe

T = TypeVar("T")


@observe.register(int)
@observe.register(float)
@observe.register(bool)
@observe.register(torch.Tensor)
def _observe_deterministic(
    rv: T, obs: Optional[AtomicObservation[T]] = None, **kwargs
) -> T:
    """
    Observe a tensor in a probabilistic program.
    """
    rv_dist = pyro.distributions.Delta(torch.as_tensor(rv), event_dim=kwargs.pop("event_dim", 0))
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
        # rv_ = pyro.sample(name, rv, obs=None, **kwargs)
        # obs = obs(rv_)
        # return observe(rv_, obs, name=name, **kwargs)
        raise NotImplementedError("Observing a distribution with a callable is not yet supported")

    if isinstance(obs, tuple):
        # for i, o in enumerate(obs):
        #     observe(rv, o, name=f"{name}[{i}]", **kwargs)
        raise NotImplementedError("Observing a distribution with a tuple is not yet supported")

    return pyro.sample(name, rv, obs=obs, **kwargs)
