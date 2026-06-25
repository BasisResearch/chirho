from __future__ import annotations

import collections
import dataclasses
import functools
from collections.abc import Collection, Hashable, Iterator, Mapping
from contextlib import contextmanager
from typing import Any, Callable, Generic, Optional, TypeVar, Union

import pyro
import torch

from chirho.counterfactual.handlers.ambiguity import FactualConditioningMessenger
from chirho.indexed.handlers import IndexPlatesMessenger
from chirho.interventional.ops import (
    AtomicIntervention,
    CompoundIntervention,
    intervene,
)
from chirho.observational.handlers.predictive import BatchedLatents
from chirho.observational.internals import unbind_leftmost_dim

K = TypeVar("K")
T = TypeVar("T")


@intervene.register(int)
@intervene.register(float)
@intervene.register(bool)
@intervene.register(torch.Tensor)
@pyro.poutine.runtime.effectful(type="intervene")
def _intervene_atom(obs, act: Optional[AtomicIntervention[T]] = None, *, event_dim: int = 0, **kwargs) -> T:
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
    obs: dict[K, T],
    act: Union[dict[K, AtomicIntervention[T]], Callable[[dict[K, T]], dict[K, T]]],
    **kwargs,
) -> dict[K, T]:
    if callable(act):
        return _dict_intervene_callable(obs, act, **kwargs)

    result: dict[K, T] = {}
    for k in obs.keys():
        result[k] = intervene(obs[k], act[k] if k in act else None, **kwargs)
    return result


@pyro.poutine.runtime.effectful(type="intervene")
def _dict_intervene_callable(obs: dict[K, T], act: Callable[[dict[K, T]], dict[K, T]], **kwargs) -> dict[K, T]:
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
        if msg["name"] not in self.actions or msg["infer"].get("_do_not_intervene", None):
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


#: Default name of the shared batch index-plate dimension.
_DEFAULT_BATCH_NAME = "batched_interventions"

#: A batch of interventions, accepted in either of two forms:
#:
#: 1. a collection of individual (``do``-style) interventions, one per scenario,
#:    mapping each intervened site to a tensor action; or
#: 2. a mapping from each site to a ``(act, mask)`` pair of tensors giving the
#:    per-index action and whether it is applied.
BatchedInterventions = Union[
    Mapping[Hashable, tuple[torch.Tensor, torch.Tensor]],
    Collection[Mapping[Hashable, torch.Tensor]],
]


@dataclasses.dataclass
class _BatchedAction:
    """A per-site intervention batched along a leading dimension.

    :param act: Action tensor of shape ``(batch_size, ...)`` giving the value to
        assign in each world.
    :param mask: Boolean tensor of shape ``(batch_size,)`` selecting, per world,
        whether ``act`` is applied (``True``) or the observed value is kept.
    """

    act: torch.Tensor
    mask: torch.Tensor

    @property
    def batch_size(self) -> int:
        """Size of the leading batch dimension."""
        return self.act.shape[0]


class _BatchedInterventions(Interventions):
    """Apply a batch of masked interventions along a shared named dimension.

    Handles :func:`~chirho.interventional.ops.intervene` for
    :class:`_BatchedAction` arguments by placing the batched ``act`` and ``mask``
    onto the named index-plate dimension ``name`` (so they align with the
    possibly-already-batched observed value regardless of event dimensions) and
    selecting between them with :func:`torch.where`.

    :param actions: Mapping from each site to its :class:`_BatchedAction`.
    :param name: Name of the shared batch index-plate dimension.
    """

    batch_size: int
    name: str

    def __init__(
        self,
        actions: Mapping[Hashable, _BatchedAction],
        name: str = _DEFAULT_BATCH_NAME,
    ):
        if not actions:
            raise ValueError("Expected a nonempty actions dict.")

        batch_sizes = {action.batch_size for action in actions.values()}
        if len(batch_sizes) != 1:
            raise ValueError("Expected each intervention to have the same batch size.")

        self.batch_size = batch_sizes.pop()
        self.name = name
        super().__init__(actions)

    def _pyro_intervene(self, msg: dict[str, Any]) -> None:
        obs, act = msg["args"]
        if not isinstance(act, _BatchedAction):
            return

        # Move the batch onto the named index-plate dimension; otherwise it lands
        # on a raw leftmost axis that neither aligns with ``obs`` nor is isolable
        # by ``gather`` (and crashes on sites with non-trivial event dimensions).
        event_dim = msg["kwargs"].get("event_dim", 0)
        act_value = unbind_leftmost_dim(act.act, self.name, size=self.batch_size, event_dim=event_dim)
        mask = unbind_leftmost_dim(
            act.mask.reshape(act.mask.shape + (1,) * event_dim),
            self.name,
            size=self.batch_size,
            event_dim=event_dim,
        )
        msg["value"] = torch.where(mask, act_value, obs)


def _prepend_factual_world(action: _BatchedAction) -> _BatchedAction:
    """Reserve index 0 of the batch axis as the factual world.

    The prepended slice carries a ``mask`` of ``False`` for every site, so the
    observed (and downstream-propagated) value passes through unchanged; its
    ``act`` entry is an unused placeholder.

    :param action: The batched action to extend with a leading factual world.
    :return: A new action whose batch dimension is one larger, with index 0
        factual and the original scenarios shifted to indices ``1..N``.
    """
    return _BatchedAction(
        torch.cat([action.act[:1], action.act], dim=0),
        torch.cat([torch.zeros_like(action.mask[:1]), action.mask], dim=0),
    )


def _build_batched_actions(interventions: BatchedInterventions, factual: bool) -> dict[Hashable, _BatchedAction]:
    """Lower either supported intervention spec to ``{site: _BatchedAction}``.

    See :class:`BatchedWorldCounterfactual` for the two accepted forms.

    :param interventions: The batch of interventions, in either accepted form.
    :param factual: Whether to prepend the factual world at index 0.
    :return: Mapping from each intervened site to its :class:`_BatchedAction`.
    """
    if isinstance(interventions, collections.abc.Mapping):
        batches = {site: _BatchedAction(*pair) for site, pair in interventions.items()}
    else:
        num_scenarios = len(interventions)
        sites = set().union(*(scenario.keys() for scenario in interventions))
        masks = {site: torch.zeros(num_scenarios, dtype=torch.bool) for site in sites}
        # ``nan`` placeholders mark slots a scenario does not intervene on; they
        # are masked out, but force a common dtype when stacked.
        acts: dict[Hashable, list[torch.Tensor]] = {
            site: [torch.tensor(float("nan"))] * num_scenarios for site in sites
        }
        for i, scenario in enumerate(interventions):
            for site, value in scenario.items():
                masks[site][i] = True
                acts[site][i] = value

        batches = {
            site: _BatchedAction(torch.stack(torch.broadcast_tensors(*values)), masks[site])
            for site, values in acts.items()
        }

    if factual:
        batches = {site: _prepend_factual_world(action) for site, action in batches.items()}

    return batches


class BatchedWorldCounterfactual(IndexPlatesMessenger):
    """Evaluate a batch of heterogeneous interventions on a single shared axis.

    Sits between :class:`~chirho.counterfactual.handlers.counterfactual.TwinWorldCounterfactual`
    (factual + one alternative) and
    :class:`~chirho.counterfactual.handlers.counterfactual.MultiWorldCounterfactual`
    (the Cartesian product of all interventions): the factual world and ``N``
    scenarios share a *single* named index-plate dimension, so memory is linear
    in ``N`` rather than exponential in the number of intervened sites. It is
    self-contained -- subclasses :class:`~chirho.indexed.handlers.IndexPlatesMessenger`,
    so needs no enclosing index-plate context -- and results are read with
    ``gather(value, IndexSet(<name>={k}), event_dim=...)``.

    ``interventions`` takes either form of :data:`BatchedInterventions`. With
    ``factual=True`` the factual world occupies index 0 and the scenarios indices
    ``1..N`` (the ``IndexSet(<name>={0})`` convention), and conditioning is
    disambiguated by :class:`~chirho.counterfactual.handlers.ambiguity.FactualConditioningMessenger`
    so observed data applies only to the factual world. With ``shared_noise=True``
    exogenous noise is held fixed across worlds, making each world exactly the
    corresponding ``MultiWorldCounterfactual`` slice; with ``shared_noise=False``
    every latent is independently batched (a batched interventional Monte Carlo).

    :param interventions: The batch of interventions, in either accepted form.
    :param name: Name of the shared batch index-plate dimension.
    :param factual: Whether to materialize the factual world at index 0.
    :param shared_noise: Whether to share exogenous noise across worlds
        (MWC-equivalent) rather than drawing it independently per scenario.
    :param first_available_dim: Leftmost dimension available for index plates.
    """

    _inner_handlers: list[pyro.poutine.messenger.Messenger]

    def __init__(
        self,
        interventions: BatchedInterventions,
        *,
        name: str = _DEFAULT_BATCH_NAME,
        factual: bool = True,
        shared_noise: bool = True,
        first_available_dim: Optional[int] = None,
    ):
        batches = _build_batched_actions(interventions, factual=factual)
        batched_interventions = _BatchedInterventions(batches, name=name)

        # Inner handlers, entered outermost-first inside this handler's index
        # plates. With shared noise the intervention itself creates the shared
        # batch axis and lets it propagate downstream (as MWC does); otherwise
        # ``BatchedLatents`` batches every latent site independently.
        self._inner_handlers = []
        if not shared_noise:
            self._inner_handlers.append(BatchedLatents(batched_interventions.batch_size, name=name))
        self._inner_handlers.append(batched_interventions)
        if factual:
            self._inner_handlers.append(FactualConditioningMessenger())

        super().__init__(first_available_dim=first_available_dim)

    def __enter__(self) -> BatchedWorldCounterfactual:
        super().__enter__()
        for handler in self._inner_handlers:
            handler.__enter__()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        for handler in reversed(self._inner_handlers):
            handler.__exit__(exc_type, exc_value, traceback)
        super().__exit__(exc_type, exc_value, traceback)


@contextmanager
def batched_do(
    interventions: BatchedInterventions,
    name: str = _DEFAULT_BATCH_NAME,
    factual: bool = True,
    shared_noise: bool = True,
) -> Iterator[None]:
    """Context-manager wrapper around :class:`BatchedWorldCounterfactual`.

    See that class for the accepted intervention forms, arguments, and semantics.
    """
    with BatchedWorldCounterfactual(interventions, name=name, factual=factual, shared_noise=shared_noise):
        yield
