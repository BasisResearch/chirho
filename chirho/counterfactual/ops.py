from __future__ import annotations

import functools
from typing import Tuple, TypeVar

import pyro

from chirho.indexed.ops import IndexSet, scatter_n
from chirho.interventional.ops import Intervention, intervene

S = TypeVar("S")
T = TypeVar("T")


@pyro.poutine.runtime.effectful(type="split")
@functools.partial(pyro.poutine.block, hide_types=["intervene"])
def split(obs: T, acts: Tuple[Intervention[T], ...], **kwargs) -> T:
    """
    Effectful primitive operation for "splitting" a combination of observational and interventional values in a
    probabilistic program into counterfactual worlds.

    :func:`~chirho.counterfactual.ops.split` returns the result of the effectful primitive operation
    :func:`~chirho.indexed.ops.scatter` applied to the concatenation of the ``obs`` and ``acts`` arguments,
    where ``obs`` represents the single observed value in the probabilistic program and ``acts`` represents
    the collection of intervention assignments.

    In a probabilistic program, :func:`split` induces a joint distribution over factual and counterfactual variables,
    where some variables are implicitly marginalized out by enclosing counterfactual handlers. For example,
    :func:`split` in the context of a :class:`~chirho.counterfactual.handlers.counterfactual.MultiWorldCounterfactual`
    handler induces a joint distribution over all combinations of ``obs`` and ``acts``, whereas
    :class:`~chirho.counterfactual.handlers.counterfactual.SingleWorldFactual` marginalizes out all ``acts``.

    :param obs: The observed value.
    :param acts: The interventions to apply.

    """
    name = kwargs.get("name", None)
    act_values = {IndexSet(**{name: {0}}): obs}
    for i, act in enumerate(acts):
        act_values[IndexSet(**{name: {i + 1}})] = intervene(obs, act, **kwargs)

    return scatter_n(act_values, event_dim=kwargs.get("event_dim", 0))
