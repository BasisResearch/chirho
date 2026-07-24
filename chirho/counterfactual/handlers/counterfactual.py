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
    """Per-site batched intervention for :class:`BatchedWorldCounterfactual`.

    :param act: Intervention values, shape ``(*batch, N, *event_shape)``.
        ``N = act.shape[-(event_dim + 1)]``.
    :param mask: Shape ``(*batch, N)``, bool or float. Selects which scenarios
        intervene on this site. Defaults to all-True.
    :param event_dim: Event dimensions trailing N in ``act``. Default 0.

    A bool mask gates: ``False`` slots return the factual value, their ``act``
    entries ignored.

    A float mask ``w`` blends: ``w * act + (1 - w) * obs``, with gradient flowing
    to both ``act`` and ``w``. Values 0 and 1 are bit-identical to the bool gate.
    The intended pattern is a straight-through estimator: a hard 0/1 mask on the
    forward pass with gradient flowing back through the soft branch to upstream
    log-probabilities. Intermediate values give a soft blend; identification results
    assuming atomic interventions no longer hold. Integer masks raise.

    All :class:`BatchedAction` objects in a single run must share the same N.
    """

    act: torch.Tensor
    mask: torch.Tensor
    event_dim: int

    def __init__(
        self,
        act: torch.Tensor,
        mask: torch.Tensor | None = None,
        *,
        event_dim: int = 0,
    ):
        self.act = act
        self.event_dim = event_dim

        n = act.shape[-(event_dim + 1)]
        if mask is None:
            batch_shape = act.shape[: act.ndim - event_dim - 1]
            self.mask = act.new_ones(batch_shape + (n,), dtype=torch.bool)
            return

        if mask.shape[-1] != n:
            raise ValueError(f"mask's last dimension must match N={n}, got mask.shape[-1]={mask.shape[-1]}.")
        if mask.dtype is not torch.bool and not mask.is_floating_point():
            raise ValueError(f"mask must be bool or floating point, got dtype {mask.dtype}.")
        self.mask = mask

    @property
    def batch_size(self) -> int:
        return self.act.shape[-(self.event_dim + 1)]


def _prepend_factual_world(action: BatchedAction) -> BatchedAction:
    """Prepend a factual world at index 0 (mask=False, so obs passes through unchanged)."""
    n_axis = action.act.ndim - action.event_dim - 1
    zero_act = torch.zeros_like(torch.narrow(action.act, n_axis, 0, 1))
    new_act = torch.cat([zero_act, action.act], dim=n_axis)
    new_mask = torch.cat([torch.zeros_like(action.mask[..., :1]), action.mask], dim=-1)
    return BatchedAction(act=new_act, mask=new_mask, event_dim=action.event_dim)


def batch_scenarios(*dicts: Mapping[Hashable, torch.Tensor]) -> dict[Hashable, BatchedAction]:
    """Convert per-scenario ``{site: tensor}`` dicts into a ``{site: BatchedAction}`` mapping.

    Infers masks from key presence. Broadcasts and stacks action tensors of differing shapes.

    :param dicts: One dict per scenario, mapping site names to intervention tensors.
    :returns: ``{site: BatchedAction}`` for use with :func:`~chirho.interventional.handlers.do`.

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
        broadcasted = list(torch.broadcast_tensors(*filled))
        stacked = torch.stack(broadcasted)
        result[site] = BatchedAction(act=stacked, mask=masks, event_dim=broadcasted[0].ndim)
    return result


class BatchedWorldCounterfactual(IndexPlatesMessenger, BaseCounterfactualMessenger):
    """Run N intervention scenarios in one vectorized pass on a single shared index-plate axis.

    Index 0 is the factual world; indices ``1..N`` are the scenarios. Read results with
    ``gather(value, IndexSet(batched_interventions={k}), event_dim=...)``.

    Pass interventions via :func:`~chirho.interventional.handlers.do` using
    :class:`BatchedAction` values. Five usage patterns:

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
        # 3 scenarios on one axis

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

    **5. Learned selection** (differentiable float mask via straight-through estimator)::

        w = torch.sigmoid(logits)              # shape (N,), requires_grad
        w = (w > 0.5).to(w) - w.detach() + w  # 0/1 forward, soft gradient
        with BatchedWorldCounterfactual():
            with do(actions={"z": BatchedAction(act=z_vals, mask=w)}):
                z, x, y = model()
        loss(y).backward()                    # logits.grad is populated

    .. note::
        Nesting inside another :class:`~chirho.indexed.handlers.IndexPlatesMessenger`
        subclass raises ``ValueError``.

    :param first_available_dim: Leftmost dimension available for index plates.
    """

    @staticmethod
    def _pyro_intervene(msg: dict[str, Any]) -> None:
        # Plain tensors/tuples fall through to the default handler and broadcast uniformly.
        if isinstance(msg["args"][1], BatchedAction):
            BaseCounterfactualMessenger._pyro_intervene(msg)

    def _pyro_split(self, msg: dict[str, Any]) -> None:
        obs, acts = msg["args"]
        if not (len(acts) == 1 and isinstance(acts[0], BatchedAction)):
            return

        action = acts[0]
        event_dim = action.event_dim

        full_action = _prepend_factual_world(action)
        batch_size = full_action.batch_size

        act_t = full_action.act.movedim(full_action.act.ndim - event_dim - 1, 0)  # (N+1, *batch, *event_shape)
        act_value = unbind_leftmost_dim(act_t, _DEFAULT_BATCH_NAME, size=batch_size, event_dim=event_dim)
        mask_t = full_action.mask.movedim(-1, 0)  # (*batch, N+1) -> (N+1, *batch)
        mask = unbind_leftmost_dim(
            mask_t.reshape(mask_t.shape + (1,) * event_dim),
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
