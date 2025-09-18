from __future__ import annotations

import collections
import dataclasses
import functools
from contextlib import contextmanager
from typing import (
    Callable,
    Collection,
    Dict,
    Generic,
    Hashable,
    Mapping,
    Optional,
    Tuple,
    TypeVar,
    Union,
)

import pyro
import torch

from chirho.interventional.ops import (
    AtomicIntervention,
    CompoundIntervention,
    intervene,
)
from chirho.observational.handlers.predictive import BatchedLatents

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
    obs: Dict[K, T],
    act: Union[Dict[K, AtomicIntervention[T]], Callable[[Dict[K, T]], Dict[K, T]]],
    **kwargs,
) -> Dict[K, T]:
    if callable(act):
        return _dict_intervene_callable(obs, act, **kwargs)

    result: Dict[K, T] = {}
    for k in obs.keys():
        result[k] = intervene(obs[k], act[k] if k in act else None, **kwargs)
    return result


@pyro.poutine.runtime.effectful(type="intervene")
def _dict_intervene_callable(
    obs: Dict[K, T], act: Callable[[Dict[K, T]], Dict[K, T]], **kwargs
) -> Dict[K, T]:
    return act(obs)


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


if isinstance(pyro.poutine.handlers._make_handler(Interventions), tuple):
    do = pyro.poutine.handlers._make_handler(Interventions)[1]
else:

    @pyro.poutine.handlers._make_handler(Interventions)
    def do(fn: Callable, actions: Mapping[Hashable, AtomicIntervention[T]]): ...


@dataclasses.dataclass
class _BatchedAction:
    act: torch.Tensor
    mask: torch.Tensor


class _BatchedInterventions(Interventions):
    def __init__(
        self, actions: Mapping[Hashable, _BatchedAction], name="batched_interventions"
    ):
        if not actions:
            raise ValueError("Expected a nonempty actions dict.")

        batch_sizes = set([v.act.shape[0] for v in actions.values()])
        if len(batch_sizes) != 1:
            raise ValueError("Expected each intervention to have the same batch size.")

        self.batch_size = list(batch_sizes)[0]

        super().__init__(actions)

    def _pyro_intervene(self, msg):
        (obs, act) = msg["args"]
        if not isinstance(act, _BatchedAction):
            return

        msg["value"] = torch.where(act.mask, act.act, obs)


@contextmanager
def batched_do(
    interventions: (
        Mapping[Hashable, Tuple[torch.Tensor, torch.Tensor]]
        | Collection[Mapping[Hashable, torch.Tensor]]
    ),
    name="batched_interventions",
):
    """Perform a batch of interventions efficiently.

    Batches can be specified either as:

    1. A collection of individual interventions, as might be passed to `do`. The
    actions are restricted to be tensors, however. The action tensors may be of
    different shapes, but they will all be broadcast together.

    2. A mapping from sample sites to pairs of tensors (act, mask) that specify
    the intervention to apply for each index in the batch and whether an
    intervention should be applied. Each act tensor should be of shape
    (batch_size, ...) and each mask tensor should be of shape (batch_size).

    .. warning:: Must be used in conjunction with :class:`~chirho.indexed.handlers.IndexPlatesMessenger` .
    """
    if isinstance(interventions, collections.abc.Mapping):
        batches = {k: _BatchedAction(*v) for (k, v) in interventions.items()}
    else:
        vars_ = set.union(*[set(i.keys()) for i in interventions])
        masks = {k: torch.zeros(len(interventions), dtype=torch.bool) for k in vars_}
        acts = {k: [torch.tensor(float("nan"))] * len(interventions) for k in vars_}
        for i, intv in enumerate(interventions):
            for k, v in intv.items():
                masks[k][i] = True
                acts[k][i] = v

        batched_acts = {
            k: torch.stack(torch.broadcast_tensors(v)) for (k, v) in intv.items()
        }
        batches = {k: _BatchedAction(batched_acts[k], masks[k]) for k in vars_}

    batched_intervene = _BatchedInterventions(batches)
    batched_latents = BatchedLatents(batched_intervene.batch_size, name=name)
    with batched_latents, batched_intervene:
        yield
