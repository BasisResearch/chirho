from __future__ import annotations

import collections
import functools
from typing import Callable, Dict, Generic, Hashable, Mapping, Optional, TypeVar

import pyro
import torch

from chirho.interventional.ops import (
    AtomicIntervention,
    CompoundIntervention,
    intervene,
)

K = TypeVar("K")
T = TypeVar("T")


@intervene.register(int)
@intervene.register(float)
@intervene.register(bool)
@intervene.register(torch.Tensor)
@pyro.poutine.runtime.effectful(type="intervene")
def _intervene_atom(
    obs, act: Optional[AtomicIntervention[T]] = None, *, event_dim: int = 0, **kwargs
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
    **kwargs,
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


@intervene.register(dict)
def _dict_intervene(
    obs: Dict[K, T], act: Dict[K, AtomicIntervention[T]], **kwargs
) -> Dict[K, T]:
    result: Dict[K, T] = {}
    for k in obs.keys():
        result[k] = intervene(obs[k], act[k] if k in act else None, **kwargs)
    return result


@intervene.register
def _intervene_callable(
    obs: collections.abc.Callable,
    act: Optional[CompoundIntervention[T]] = None,
    **call_kwargs,
) -> Callable[..., T]:
    if act is None:
        return obs
    elif callable(act):

        @functools.wraps(obs)
        def _intervene_callable_wrapper(*args, **kwargs):
            return intervene(obs(*args, **kwargs), act(*args, **kwargs), **call_kwargs)

        return _intervene_callable_wrapper
    return Interventions(actions=act)(obs)


class Interventions(Generic[T], pyro.poutine.messenger.Messenger):
    """
    Intervene on values in a probabilistic program.

    :class:`DoMessenger` is an effect handler that intervenes at specified sample sites
    in a probabilistic program. This allows users to define programs without any
    interventional or causal semantics, and then to add those features later in the
    context of, for example, :class:`DoMessenger`. This handler uses :func:`intervene`
    internally and supports the same types of interventions.
    """

    def __init__(self, actions: Mapping[Hashable, AtomicIntervention[T]]):
        """
        :param actions: A mapping from names of sample sites to interventions.
        """
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


do = pyro.poutine.handlers._make_handler(Interventions)[1]
