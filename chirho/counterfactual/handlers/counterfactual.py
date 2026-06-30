from __future__ import annotations

import collections
import dataclasses
from collections.abc import Collection, Hashable, Iterator, Mapping
from contextlib import ExitStack, contextmanager
from typing import Any, Optional, TypeVar, Union

import pyro
import torch

from chirho.counterfactual.handlers.ambiguity import FactualConditioningMessenger
from chirho.counterfactual.ops import split
from chirho.indexed.handlers import IndexPlatesMessenger
from chirho.indexed.ops import get_index_plates
from chirho.interventional.handlers import Interventions
from chirho.interventional.ops import intervene
from chirho.observational.handlers.predictive import BatchedLatents
from chirho.observational.internals import unbind_leftmost_dim

T = TypeVar("T")


class BaseCounterfactualMessenger(FactualConditioningMessenger):
    """
    Base class for counterfactual handlers.

    :class:`~chirho.counterfactual.handlers.counterfactual.BaseCounterfactualMessenger` is an effect handler
    for imbuing :func:`~chirho.interventional.ops.intervene` operations with world-splitting
    semantics that is useful for downstream causal and counterfactual reasoning. Specifically,
    :class:`~chirho.counterfactual.handlers.counterfactual.BaseCounterfactualMessenger` handles
    :func:`~chirho.interventional.ops.intervene` by instantiating the primitive operation
    :func:`~chirho.counterfactual.ops.split`, which is then subsequently handled by subclasses
    such as :class:`~chirho.counterfactual.handlers.counterfactual.MultiWorldCounterfactual`.
    """

    @staticmethod
    def _pyro_intervene(msg: dict[str, Any]) -> None:
        msg["stop"] = True
        if msg["args"][1] is not None:
            obs, acts = msg["args"][0], msg["args"][1]
            acts = acts(obs) if callable(acts) else acts
            acts = (acts,) if not isinstance(acts, tuple) else acts
            msg["value"] = split(obs, acts, name=msg["name"], **msg["kwargs"])
            msg["done"] = True

    @staticmethod
    def _pyro_preempt(msg: dict[str, Any]) -> None:
        if msg["kwargs"].get("name", None) is None:
            msg["kwargs"]["name"] = msg["name"]


class SingleWorldCounterfactual(BaseCounterfactualMessenger):
    """
    Trivial counterfactual handler that returns the intervened value.

    :class:`~chirho.counterfactual.handlers.counterfactual.SingleWorldCounterfactual` is an effect handler
    that subclasses :class:`~chirho.counterfactual.handlers.counterfactual.BaseCounterfactualMessenger` and
    handles :func:`~chirho.counterfactual.ops.split` primitive operations. See the documentation for
    :func:`~chirho.counterfactual.ops.split` for more details about the interaction between the enclosing
    counterfactual handler and the induced joint marginal distribution over factual and counterfactual variables.

    :class:`~chirho.counterfactual.handlers.counterfactual.SingleWorldCounterfactual` handles
    :func:`~chirho.counterfactual.ops.split` by returning only the final element in the collection
    of intervention assignments ``acts``, ignoring all other intervention assignments and observed values ``obs``.
    This can be thought of as marginalizing out all of the factual and counterfactual variables except for the
    counterfactual induced by the final element in the collection of intervention assignments in the probabilistic
    program. ::

        >>> with SingleWorldCounterfactual():
        ...     x = torch.tensor(1.)
        ...     x = intervene(x, torch.tensor(0.))
        >>> assert (x == torch.tensor(0.))
    """

    @pyro.poutine.block(hide_types=["intervene"])
    def _pyro_split(self, msg: dict[str, Any]) -> None:
        obs, acts = msg["args"]
        msg["value"] = intervene(obs, acts[-1], **msg["kwargs"])
        msg["done"] = True
        msg["stop"] = True


class SingleWorldFactual(BaseCounterfactualMessenger):
    """
    Trivial counterfactual handler that returns the observed value.

    :class:`~chirho.counterfactual.handlers.counterfactual.SingleWorldFactual` is an effect handler
    that subclasses :class:`~chirho.counterfactual.handlers.counterfactual.BaseCounterfactualMessenger` and
    handles :func:`~chirho.counterfactual.ops.split` primitive operations. See the documentation for
    :func:`~chirho.counterfactual.ops.split` for more details about the interaction between the enclosing
    counterfactual handler and the induced joint marginal distribution over factual and counterfactual variables.

    :class:`~chirho.counterfactual.handlers.counterfactual.SingleWorldFactual` handles
    :func:`~chirho.counterfactual.ops.split` by returning only the observed value ``obs``,
    ignoring all intervention assignments ``act``. This can be thought of as marginalizing out
    all of the counterfactual variables in the probabilistic program. ::

        >>> with SingleWorldFactual():
        ...    x = torch.tensor(1.)
        ...    x = intervene(x, torch.tensor(0.))
        >>> assert (x == torch.tensor(1.))
    """

    @staticmethod
    def _pyro_split(msg: dict[str, Any]) -> None:
        obs, _ = msg["args"]
        msg["value"] = obs
        msg["done"] = True
        msg["stop"] = True


class MultiWorldCounterfactual(IndexPlatesMessenger, BaseCounterfactualMessenger):
    """
    Counterfactual handler that returns all observed and intervened values.

    :class:`~chirho.counterfactual.handlers.counterfactual.MultiWorldCounterfactual` is an effect handler
    that subclasses :class:`~chirho.indexed.handlers.IndexPlatesMessenger` and
    :class:`~chirho.counterfactual.handlers.counterfactual.BaseCounterfactualMessenger` base classes.


    .. note:: Handlers that subclass :class:`~chirho.indexed.handlers.IndexPlatesMessenger` such as
       :class:`~chirho.counterfactual.handlers.counterfactual.MultiWorldCounterfactual` return tensors that can
       be cumbersome to index into directly. Therefore, we strongly recommend using ``chirho``'s indexing operations
       :func:`~chirho.indexed.ops.gather` and :class:`~chirho.indexed.ops.IndexSet` whenever using
       :class:`~chirho.counterfactual.handlers.counterfactual.MultiWorldCounterfactual` handlers.

    :class:`~chirho.counterfactual.handlers.counterfactual.MultiWorldCounterfactual`
    handles :func:`~chirho.counterfactual.ops.split` primitive operations. See the documentation for
    :func:`~chirho.counterfactual.ops.split` for more details about the interaction between the enclosing
    counterfactual handler and the induced joint marginal distribution over factual and counterfactual variables.

    :class:`~chirho.counterfactual.handlers.counterfactual.MultiWorldCounterfactual` handles
    :func:`~chirho.counterfactual.ops.split` by returning all observed values ``obs`` and intervened values ``act``.
    This can be thought of as returning the full joint distribution over all factual and counterfactual variables. ::

        >>> with MultiWorldCounterfactual():
        ...    x = torch.tensor(1.)
        ...    x = intervene(x, torch.tensor(0.), name="x_ax_1")
        ...    x = intervene(x, torch.tensor(2.), name="x_ax_2")
        ...    x_factual = gather(x, IndexSet(x_ax_1={0}, x_ax_2={0}))
        ...    x_counterfactual_1 = gather(x, IndexSet(x_ax_1={1}, x_ax_2={0}))
        ...    x_counterfactual_2 = gather(x, IndexSet(x_ax_1={0}, x_ax_2={1}))

        >>> assert(x_factual.squeeze() == torch.tensor(1.))
        >>> assert(x_counterfactual_1.squeeze() == torch.tensor(0.))
        >>> assert(x_counterfactual_2.squeeze() == torch.tensor(2.))
    """

    fresh_prefix: str = "__fresh_split__"

    @classmethod
    def _pyro_split(cls, msg: dict[str, Any]) -> None:
        if msg["name"] is None:
            index_plates = get_index_plates()
            name, fresh_suffix = cls.fresh_prefix, len(index_plates)
            while name in index_plates:
                name = f"{cls.fresh_prefix}{fresh_suffix}"
                fresh_suffix += 1
        else:
            name = msg["name"]
        msg["kwargs"]["name"] = msg["name"] = name


class TwinWorldCounterfactual(IndexPlatesMessenger, BaseCounterfactualMessenger):
    """
    Counterfactual handler that returns all observed values and the final intervened value.

    :class:`~chirho.counterfactual.handlers.counterfactual.TwinWorldCounterfactual` is an effect handler
    that subclasses :class:`~chirho.indexed.handlers.IndexPlatesMessenger` and
    :class:`~chirho.counterfactual.handlers.counterfactual.BaseCounterfactualMessenger` base classes.


    .. note:: Handlers that subclass :class:`~chirho.indexed.handlers.IndexPlatesMessenger` such as
       :class:`~chirho.counterfactual.handlers.counterfactual.TwinWorldCounterfactual` return tensors that can
       be cumbersome to index into directly. Therefore, we strongly recommend using ``chirho``'s indexing operations
       :func:`~chirho.indexed.ops.gather` and :class:`~chirho.indexed.ops.IndexSet` whenever using
       :class:`~chirho.counterfactual.handlers.counterfactual.TwinWorldCounterfactual` handlers.

    :class:`~chirho.counterfactual.handlers.counterfactual.TwinWorldCounterfactual`
    handles :func:`~chirho.counterfactual.ops.split` primitive operations. See the documentation for
    :func:`~chirho.counterfactual.ops.split` for more details about the interaction between the enclosing
    counterfactual handler and the induced joint marginal distribution over factual and counterfactual variables.

    :class:`~chirho.counterfactual.handlers.counterfactual.TwinWorldCounterfactual` handles
    :func:`~chirho.counterfactual.ops.split` by returning the observed values ``obs`` and the
    final intervened values ``act`` in the probabilistic program. This can be thought of as returning
    the joint distribution over factual and counterfactual variables, marginalizing out all but the final
    configuration of intervention assignments in the probabilistic program. ::

        >>> with TwinWorldCounterfactual():
        ...    x = torch.tensor(1.)
        ...    x = intervene(x, torch.tensor(0.))
        ...    x = intervene(x, torch.tensor(2.))
        >>> # TwinWorldCounterfactual ignores the first intervention
        >>> assert(x.squeeze().shape == torch.Size([2]))
        >>> assert(x.squeeze()[0] == torch.tensor(1.))
        >>> assert(x.squeeze()[1] == torch.tensor(2.))
    """

    fresh_prefix: str = "__fresh_split__"

    @classmethod
    def _pyro_split(cls, msg: dict[str, Any]) -> None:
        msg["kwargs"]["name"] = msg["name"] = cls.fresh_prefix


_DEFAULT_BATCH_NAME = "batched_interventions"

#: Either a collection of per-scenario ``{site: tensor}`` dicts, or a mapping from
#: each site to an ``(act, mask)`` pair of tensors with a shared leading batch dimension.
BatchedInterventions = Union[
    Mapping[Hashable, tuple[torch.Tensor, torch.Tensor]],
    Collection[Mapping[Hashable, torch.Tensor]],
]


@dataclasses.dataclass
class _BatchedAction:
    """Per-site intervention batched along a leading dimension.

    For N scenarios and a site with event shape ``(E,)``:

    - ``act`` has shape ``(N, E)`` — the value to assign in each scenario.
    - ``mask`` has shape ``(N,)`` — ``True`` where the scenario intervenes on
      this site, ``False`` where the observed value should pass through unchanged.

    ``act`` must cover all N slots even when some scenarios do not intervene on
    this site, because ``torch.where(mask, act, obs)`` requires both branches to
    be full tensors of the same shape. Slots where ``mask`` is ``False`` are
    never selected; their values are placeholders (zeros) and are discarded.

    For a site with event shape ``(4,)`` and N=3 scenarios where only the
    first two intervene::

        act  = torch.zeros(3, 4)                  # shape (3, 4)
        act[0] = torch.tensor([1., 2., 3., 4.])   # scenario 0 intervenes
        act[1] = torch.tensor([5., 6., 7., 8.])   # scenario 1 intervenes
        # act[2] is a zero placeholder: scenario 2 may intervene on a different
        # site (e.g. z) but not on this one. All N scenarios share one axis, so
        # every site's act tensor has N rows regardless. mask[2]=False ensures
        # torch.where never reads this slot.
        mask = torch.tensor([True, True, False])  # shape (3,); no event dims
    """

    act: torch.Tensor
    mask: torch.Tensor

    def __post_init__(self) -> None:
        if self.act.shape[0] != self.mask.shape[0]:
            raise ValueError(
                f"act and mask must have the same leading dimension, "
                f"got act.shape[0]={self.act.shape[0]} and mask.shape[0]={self.mask.shape[0]}."
            )

    @property
    def batch_size(self) -> int:
        return self.act.shape[0]


class _BatchedInterventions(Interventions):
    """Handles :func:`~chirho.interventional.ops.intervene` for :class:`_BatchedAction`
    arguments, placing ``act`` and ``mask`` onto the named index-plate dimension and
    selecting between them with :func:`torch.where`.
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
    """Prepend a factual world at index 0 (mask=False, so obs passes through unchanged)."""
    return _BatchedAction(
        torch.cat([torch.zeros_like(action.act[:1]), action.act], dim=0),
        torch.cat([torch.zeros_like(action.mask[:1]), action.mask], dim=0),
    )


def _build_batched_actions(interventions: BatchedInterventions, factual: bool) -> dict[Hashable, _BatchedAction]:
    """Convert either accepted intervention form to ``{site: _BatchedAction}``.

    Accepts two forms.

    Mapping form — the caller supplies ``(act, mask)`` pairs directly::

        interventions = {
            "x": (torch.tensor([10.0, -5.0]), torch.tensor([True, True])),
            "z": (torch.tensor([0.0,  99.0]), torch.tensor([False, True])),
        }

    Collection form — a list of per-scenario dicts, one entry per scenario that
    actually intervenes::

        interventions = [
            {"x": torch.tensor(10.0)},           # scenario 0: intervene on x only
            {"x": torch.tensor(-5.0), "z": torch.tensor(99.0)},  # scenario 1: both
        ]

    Both forms produce the same ``{site: _BatchedAction}`` output. For the
    collection form above with N=2 scenarios and sites ``{"x", "z"}``, the result
    is::

        "x": _BatchedAction(
            act  = torch.tensor([10.0, -5.0]),   # shape (2,)
            mask = torch.tensor([True,  True]),
        )
        "z": _BatchedAction(
            act  = torch.tensor([0.0,  99.0]),   # shape (2,); slot 0 is a placeholder
            mask = torch.tensor([False, True]),   # scenario 0 doesn't touch z
        )

    Action tensors for different scenarios may have different shapes; they are
    aligned with ``torch.broadcast_tensors`` before stacking, so a scalar
    intervention broadcasts to match a vector one.

    If ``factual=True``, an extra slot is prepended at index 0 with ``mask=False``
    for every site, reserving it as the unintervened factual world.
    """
    if isinstance(interventions, collections.abc.Mapping):
        batches = {site: _BatchedAction(*pair) for site, pair in interventions.items()}
    else:
        num_scenarios = len(interventions)
        sites = set().union(*(scenario.keys() for scenario in interventions))
        masks = {site: torch.zeros(num_scenarios, dtype=torch.bool) for site in sites}
        acts: dict[Hashable, list[Optional[torch.Tensor]]] = {site: [None] * num_scenarios for site in sites}
        for i, scenario in enumerate(interventions):
            for site, value in scenario.items():
                masks[site][i] = True
                acts[site][i] = value

        # Slots that no scenario intervenes on are masked out (mask=False) and
        # never selected by torch.where, so their value is irrelevant.  We fill
        # them with a scalar zero of the site's dtype so that torch.stack gets a
        # uniform dtype without silently upcasting integer-typed interventions to
        # float.  Every site has at least one real entry (it entered ``sites``
        # through some scenario).
        for site, values in acts.items():
            real = next(v for v in values if v is not None)
            acts[site] = [torch.zeros((), dtype=real.dtype) if v is None else v for v in values]

        batches = {
            site: _BatchedAction(torch.stack(torch.broadcast_tensors(*values)), masks[site])  # type: ignore[arg-type]
            for site, values in acts.items()
        }

    if factual:
        batches = {site: _prepend_factual_world(action) for site, action in batches.items()}

    return batches


class BatchedWorldCounterfactual(IndexPlatesMessenger):
    """Evaluate a batch of heterogeneous interventions on a single shared axis.

    Sits between :class:`~chirho.counterfactual.handlers.counterfactual.TwinWorldCounterfactual`
    (one alternative) and
    :class:`~chirho.counterfactual.handlers.counterfactual.MultiWorldCounterfactual`
    (Cartesian product): N scenarios share one named index-plate dimension, so
    memory is linear in N rather than exponential in the number of intervened sites.
    Results are read with ``gather(value, IndexSet(<name>={k}), event_dim=...)``.

    ``interventions`` accepts either form of :data:`BatchedInterventions`.

    With ``factual=True`` (default) the factual world occupies index 0 and
    scenarios indices ``1..N``; observed data is applied only to the factual world.
    With ``shared_noise=True`` (default) exogenous noise is shared across worlds
    (equivalent to the corresponding :class:`MultiWorldCounterfactual` slice);
    with ``shared_noise=False`` every latent is independently batched.

    :param interventions: Batch of interventions in either accepted form.
    :param factual: Materialize the factual world at index 0.
    :param shared_noise: Share exogenous noise across worlds.
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
        self._exit_stack = ExitStack()
        self._exit_stack.__enter__()
        for handler in self._inner_handlers:
            self._exit_stack.enter_context(handler)
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        try:
            self._exit_stack.__exit__(exc_type, exc_value, traceback)
        finally:
            super().__exit__(exc_type, exc_value, traceback)


@contextmanager
def batched_do(
    interventions: BatchedInterventions,
    name: str = _DEFAULT_BATCH_NAME,
    factual: bool = True,
    shared_noise: bool = True,
    first_available_dim: Optional[int] = None,
) -> Iterator[None]:
    """Context manager form of :class:`BatchedWorldCounterfactual`."""
    with BatchedWorldCounterfactual(
        interventions,
        name=name,
        factual=factual,
        shared_noise=shared_noise,
        first_available_dim=first_available_dim,
    ):
        yield
