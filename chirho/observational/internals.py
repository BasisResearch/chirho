from typing import Optional, TypeVar

import pyro
import pyro.distributions
import torch

from chirho.observational.ops import AtomicObservation, observe

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


class ObserveNameMessenger(pyro.poutine.messenger.Messenger):
    def _pyro_observe(self, msg):
        if "name" not in msg["kwargs"]:
            msg["kwargs"]["name"] = msg["name"]
