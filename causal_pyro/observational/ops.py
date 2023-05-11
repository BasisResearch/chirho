import functools
from typing import Callable, Hashable, Mapping, Optional, Tuple, TypeVar, Union

T = TypeVar("T")

AtomicObservation = Union[T, Tuple[T, ...], Callable[[T], Union[T, Tuple[T, ...]]]]
CompoundObservation = Union[Mapping[Hashable, AtomicObservation[T]], Callable[..., T]]
Observation = Union[AtomicObservation[T], CompoundObservation[T]]


@functools.singledispatch
def observe(rv, obs: Optional[Observation[T]] = None, **kwargs):
    """
    Observe a random value in a probabilistic program.
    """
    raise NotImplementedError(f"observe not implemented for type {type(rv)}")


@observe.register(int)
@observe.register(float)
@observe.register(bool)
@observe.register(torch.Tensor)
def _observe_deterministic(rv: T, obs: Optional[AtomicObservation[T]] = None, **kwargs) -> T:
    """
    Observe a tensor in a probabilistic program.
    """
    if obs is None:
        return rv
    rv_dist = pyro.distributions.Delta(rv, event_dim=kwargs.pop("event_dim", 0))
    return pyro.sample("rv", observe(rv_dist, obs, **kwargs))


@observe.register(pyro.distributions.Distribution)
@pyro.poutine.runtime.effectful(type="observe")
def _observe_distribution(
    rv: pyro.distributions.Distribution,
    obs: Optional[pyro.distributions.Distribution] = None,
    **kwargs
) -> pyro.distributions.Distribution:
    if obs is None:
        return rv
    obs_val = pyro.sample("obs", obs)
    log_ratio = rv.log_prob(obs_val) - obs.log_prob(obs_val)
    pyro.factor("rv", log_ratio)
    return pyro.distributions.Delta(obs_val, event_dim=rv.event_dim)
