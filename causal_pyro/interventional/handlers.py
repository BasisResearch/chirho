import collections
import functools
from typing import Callable, Generic, Hashable, Mapping, Optional, TypeVar

import pyro
import torch

from causal_pyro.interventional.ops import (
    AtomicIntervention,
    CompoundIntervention,
    intervene,
)

T = TypeVar("T")


@intervene.register(int)
@intervene.register(float)
@intervene.register(bool)
@intervene.register(torch.Tensor)
@pyro.poutine.runtime.effectful(type="intervene")
def _intervene_atom(
    obs: T, act: Optional[AtomicIntervention[T]] = None, *, event_dim: int = 0, **kwargs
) -> T:
    """
    Intervene on an atomic value in a probabilistic program.
    """
    if act is None:
        return obs
    elif callable(act):
        act = act(obs)
        return act[-1] if isinstance(act, tuple) else act
    elif isinstance(act, tuple):
        return act[-1]
    return act


@intervene.register(pyro.distributions.Distribution)
@pyro.poutine.runtime.effectful(type="intervene")
def _intervene_atom_distribution(
    obs: pyro.distributions.Distribution,
    act: Optional[AtomicIntervention[pyro.distributions.Distribution]] = None,
    **kwargs
) -> pyro.distributions.Distribution:
    """
    Intervene on a distribution in a probabilistic program.
    """
    if act is None:
        return obs
    elif callable(act) and not isinstance(act, pyro.distributions.Distribution):
        act = act(obs)
        return act[-1] if isinstance(act, tuple) else act
    elif isinstance(act, tuple):
        return act[-1]
    return act


@intervene.register
def _intervene_callable(
    obs: collections.abc.Callable,
    act: Optional[CompoundIntervention[T]] = None,
    **call_kwargs
) -> Callable[..., T]:
    if act is None:
        return obs
    elif callable(act):

        @functools.wraps(obs)
        def _intervene_callable_wrapper(*args, **kwargs):
            return intervene(obs(*args, **kwargs), act(*args, **kwargs), **call_kwargs)

        return _intervene_callable_wrapper
    return DoMessenger(actions=act)(obs)


class DoMessenger(Generic[T], pyro.poutine.messenger.Messenger):
    """
    Intervene on values in a probabilistic program.
    """

    def __init__(self, actions: Mapping[Hashable, AtomicIntervention[T]]):
        self.actions = actions
        super().__init__()

    def _pyro_post_sample(self, msg):
        if msg["name"] not in self.actions or msg["infer"].get(
            "_do_not_intervene", None
        ):
            return

        msg["value"] = intervene(
            msg["value"],
            self.actions[msg["name"]],
            event_dim=len(msg["fn"].event_shape),
            name=msg["name"],
        )


do = pyro.poutine.handlers._make_handler(DoMessenger)[1]
