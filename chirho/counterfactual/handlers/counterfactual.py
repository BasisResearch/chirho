from __future__ import annotations

import dataclasses
from collections.abc import Hashable, Mapping
from typing import Any, Optional, TypeVar

import pyro
import torch

from chirho.counterfactual.handlers.ambiguity import FactualConditioningMessenger
from chirho.counterfactual.ops import split
from chirho.indexed.handlers import IndexPlatesMessenger
from chirho.indexed.ops import get_index_plates
from chirho.interventional.ops import intervene
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


@dataclasses.dataclass
class BatchedAction:
    """Per-site batched intervention for use with :class:`BatchedWorldCounterfactual`.

    :param act: Shape ``(N, *event_shape)``. Intervention value for each scenario.
    :param mask: Shape ``(N,)``, bool or floating point. Which scenarios intervene on
        this site. Defaults to all-True.
    :param validate_args: If True, check that a floating point ``mask`` lies in ``[0, 1]``.

    A bool ``mask`` gates. ``torch.where`` never reads slots where it is ``False``,
    so their ``act`` values are irrelevant placeholders.

    A floating point ``mask`` ``w`` blends, ``w * act + (1 - w) * obs``, and passes
    gradient to both ``act`` and ``w``. Autograd rejects ``requires_grad`` on bool, so
    a floating point dtype is the only way to learn a mask. Values of exactly 0 or 1
    match the bool path bit for bit; that is the intended use, typically via a
    straight-through estimator. Intermediate values give a soft intervention, and
    identification results that assume atomic interventions no longer hold. Values
    outside ``[0, 1]`` extrapolate. The blend multiplies both branches instead of
    picking one, so it requires a floating point site and finite ``act`` and ``obs``.
    Integer masks raise.

    All ``BatchedAction`` objects in a single run must share the same leading size N.
    """

    act: torch.Tensor
    mask: torch.Tensor

    def __init__(
        self,
        act: torch.Tensor,
        mask: torch.Tensor | None = None,
        *,
        validate_args: bool = False,
    ):
        self.act = act
        if mask is None:
            self.mask = torch.ones(self.act.shape[0], dtype=torch.bool)
            return

        if self.act.shape[0] != mask.shape[0]:
            raise ValueError(
                f"act and mask must have the same leading dimension, "
                f"got act.shape[0]={self.act.shape[0]} and mask.shape[0]={mask.shape[0]}."
            )
        if mask.dtype is not torch.bool and not mask.is_floating_point():
            raise ValueError(f"mask must be bool or floating point, got dtype {mask.dtype}.")
        if validate_args and mask.is_floating_point() and not ((mask >= 0.0) & (mask <= 1.0)).all():
            raise ValueError("floating point mask must lie in [0, 1].")
        self.mask = mask

    @property
    def batch_size(self) -> int:
        return self.act.shape[0]


def _prepend_factual_world(action: BatchedAction) -> BatchedAction:
    """Prepend a factual world at index 0 (mask=False, so obs passes through unchanged)."""
    assert action.mask is not None
    return BatchedAction(
        torch.cat([torch.zeros_like(action.act[:1]), action.act], dim=0),
        torch.cat([torch.zeros_like(action.mask[:1]), action.mask], dim=0),
    )


def batch_scenarios(*dicts: Mapping[Hashable, torch.Tensor]) -> dict[Hashable, BatchedAction]:
    """Convert per-scenario ``{site: tensor}`` dicts to a ``{site: BatchedAction}`` mapping.

    Masks are inferred from key presence. Action tensors of different shapes are broadcast
    before stacking, so a scalar intervention broadcasts to match a vector one.

    :param dicts: One dict per scenario mapping site names to intervention tensors.
    :returns: ``{site: BatchedAction}`` suitable for passing to
        :func:`~chirho.interventional.handlers.do`.

    Example::

        actions = batch_scenarios(
            {"z": torch.tensor(1.0)},                        # scenario 0: z only
            {"z": torch.tensor(2.0), "x": torch.tensor(3.0)},  # scenario 1: z and x
            {"x": torch.tensor(3.0)},                        # scenario 2: x only
        )
        with BatchedWorldCounterfactual():
            with do(actions=actions):
                model()
    """
    if not dicts:
        raise ValueError("batch_scenarios requires at least one scenario dict.")
    sites: set[Hashable] = set().union(*(d.keys() for d in dicts))
    result: dict[Hashable, BatchedAction] = {}
    for site in sites:
        raw: list[Optional[torch.Tensor]] = [d.get(site) for d in dicts]  # type: ignore[arg-type]
        masks = torch.tensor([v is not None for v in raw])
        real = next(v for v in raw if v is not None)
        filled = [torch.zeros((), dtype=real.dtype) if v is None else v for v in raw]  # type: ignore[union-attr]
        stacked = torch.stack(list(torch.broadcast_tensors(*filled)))
        result[site] = BatchedAction(act=stacked, mask=masks)
    return result


class BatchedWorldCounterfactual(IndexPlatesMessenger, BaseCounterfactualMessenger):
    """Evaluate N heterogeneous intervention scenarios on a single shared index-plate axis.

    Memory is linear in N (vs :class:`MultiWorldCounterfactual`'s exponential in sites).
    Index 0 is the factual world; indices ``1..N`` are the scenarios.  Results are read with
    ``gather(value, IndexSet(batched_interventions={k}), event_dim=...)``.

    Interventions are passed via :func:`~chirho.interventional.handlers.do` using
    :class:`BatchedAction` values.  The five main usage patterns:

    **1. Sweep one site over N values** (no mask needed)::

        z_vals = torch.linspace(-3.0, 3.0, 5)  # shape (5,)
        with BatchedWorldCounterfactual():
            with do(actions={"z": BatchedAction(act=z_vals)}):
                z, x, y = model()
        # world 0: factual; worlds 1-5: z fixed at each value

    **2. Heterogeneous per-site masks** (each scenario intervenes on a different subset)::

        actions = {
            "z": BatchedAction(act=torch.tensor([1., 2., 0.]), mask=torch.tensor([True, True, False])),
            "x": BatchedAction(act=torch.tensor([0., 3., 3.]), mask=torch.tensor([False, True, True])),
        }
        with BatchedWorldCounterfactual():
            with do(actions=actions):
                z, x, y = model()
        # 3 scenarios share one axis instead of MWC's 2x2 cross-product

    **3. Scenario dicts via** :func:`batch_scenarios` (masks inferred from key presence)::

        with BatchedWorldCounterfactual():
            with do(actions=batch_scenarios(
                {"z": torch.tensor(1.)},
                {"z": torch.tensor(2.), "x": torch.tensor(3.)},
                {"x": torch.tensor(3.)},
            )):
                z, x, y = model()

    **4. PCI — pre-sampled masks and values** (sufficiency + necessity in one pass)::

        N = 4
        actions = {
            site: BatchedAction(
                act=torch.cat([suff_vals[site], nec_vals[site]]),
                mask=torch.cat([masks[site], masks[site]]),
            )
            for site in sites
        }
        with BatchedWorldCounterfactual():
            with do(actions=actions):
                z, x, y = model()
        # world 0: factual; worlds 1..N: sufficiency; worlds N+1..2N: necessity

    **5. Learned selection** (float mask, differentiable in both ``act`` and ``mask``)::

        w = torch.sigmoid(logits)             # requires_grad, shape (N,)
        w = (w > 0.5).to(w) - w.detach() + w  # straight-through: k-hot forward, soft gradient
        with BatchedWorldCounterfactual():
            with do(actions={"z": BatchedAction(act=z_vals, mask=w)}):
                z, x, y = model()
        loss(y).backward()                    # logits.grad is populated

    .. note::
        Nesting inside another :class:`~chirho.indexed.handlers.IndexPlatesMessenger`
        subclass (e.g. :class:`MultiWorldCounterfactual`) is not currently supported
        and raises ``ValueError`` — the same pre-existing limitation as nesting MWC
        inside :class:`TwinWorldCounterfactual`.

    :param first_available_dim: Leftmost dimension available for index plates.
    """

    @staticmethod
    def _pyro_intervene(msg: dict[str, Any]) -> None:
        # Only convert BatchedAction interventions to split.  Plain tensors and
        # tuples are returned as-is (the default _intervene_atom body runs) so
        # that a vanilla do() inside BatchedWorldCounterfactual broadcasts its
        # value across the existing batched_interventions axis rather than
        # creating a new per-site plate.
        if isinstance(msg["args"][1], BatchedAction):
            BaseCounterfactualMessenger._pyro_intervene(msg)

    def _pyro_split(self, msg: dict[str, Any]) -> None:
        obs, acts = msg["args"]
        if not (len(acts) == 1 and isinstance(acts[0], BatchedAction)):
            return

        action = acts[0]
        event_dim = msg["kwargs"].get("event_dim", 0)

        full_action = _prepend_factual_world(action)
        batch_size = full_action.batch_size

        act_value = unbind_leftmost_dim(full_action.act, _DEFAULT_BATCH_NAME, size=batch_size, event_dim=event_dim)
        mask = unbind_leftmost_dim(
            full_action.mask.reshape(full_action.mask.shape + (1,) * event_dim),
            _DEFAULT_BATCH_NAME,
            size=batch_size,
            event_dim=event_dim,
        )
        if mask.dtype is torch.bool:
            msg["value"] = torch.where(mask, act_value, obs)
        else:
            if not obs.is_floating_point():
                raise ValueError(
                    f"a floating point BatchedAction.mask blends against the site value, which "
                    f"requires a floating point site, but {msg['kwargs'].get('name')!r} has dtype "
                    f"{obs.dtype}. Use a bool mask for discrete sites."
                )
            w = mask.to(obs.dtype)
            msg["value"] = w * act_value + (1 - w) * obs
        msg["done"] = True
        msg["stop"] = True
