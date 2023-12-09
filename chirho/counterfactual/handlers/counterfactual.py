from __future__ import annotations

from typing import Any, Dict, Generic, Mapping, TypeVar

import pyro
import torch

from chirho.counterfactual.handlers.ambiguity import FactualConditioningMessenger
from chirho.counterfactual.ops import preempt, split
from chirho.indexed.handlers import IndexPlatesMessenger
from chirho.indexed.ops import get_index_plates
from chirho.interventional.ops import Intervention, intervene

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

    In addition, :class:`~chirho.counterfactual.handlers.counterfactual.BaseCounterfactualMessenger`
    handles :func:`~chirho.counterfactual.ops.preempt` operations by introducing an auxiliary categorical
    variable at each of the preempted addresses.
    """

    @staticmethod
    def _pyro_intervene(msg: Dict[str, Any]) -> None:
        msg["stop"] = True
        if msg["args"][1] is not None:
            obs, acts = msg["args"][0], msg["args"][1]
            acts = acts(obs) if callable(acts) else acts
            acts = (acts,) if not isinstance(acts, tuple) else acts
            msg["value"] = split(obs, acts, name=msg["name"], **msg["kwargs"])
            msg["done"] = True

    @staticmethod
    def _pyro_preempt(msg: Dict[str, Any]) -> None:
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
    def _pyro_split(self, msg: Dict[str, Any]) -> None:
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
    def _pyro_split(msg: Dict[str, Any]) -> None:
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

    default_name: str = "intervened"

    @classmethod
    def _pyro_split(cls, msg: Dict[str, Any]) -> None:
        name = msg["name"] if msg["name"] is not None else cls.default_name
        index_plates = get_index_plates()
        if name in index_plates:
            name = f"{name}__dup_{len(index_plates)}"
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

    default_name: str = "intervened"

    @classmethod
    def _pyro_split(cls, msg: Dict[str, Any]) -> None:
        msg["kwargs"]["name"] = msg["name"] = cls.default_name


class Preemptions(Generic[T], pyro.poutine.messenger.Messenger):
    """
    Effect handler that applies the operation :func:`~chirho.counterfactual.ops.preempt`
    to sample sites in a probabilistic program,
    similar to the handler :func:`~chirho.observational.handlers.condition`
    for :func:`~chirho.observational.ops.observe` .
    or the handler :func:`~chirho.interventional.handlers.do`
    for :func:`~chirho.interventional.ops.intervene` .

    See the documentation for :func:`~chirho.counterfactual.ops.preempt` for more details.

    This handler introduces an auxiliary discrete random variable at each preempted sample site
    whose name is the name of the sample site prefixed by ``prefix``, and
    whose value is used as the ``case`` argument to :func:`preempt`,
    to determine whether the preemption returns the present value of the site
    or the new value specified for the site in ``actions``

    The distributions of the auxiliary discrete random variables are parameterized by ``bias``.
    By default, ``bias == 0`` and the value returned by the sample site is equally likely
    to be the factual case (i.e. the present value of the site) or one of the counterfactual cases
    (i.e. the new value(s) specified for the site in ``actions``).
    When ``0 < bias <= 0.5``, the preemption is less than equally likely to occur.
    When ``-0.5 <= bias < 0``, the preemption is more than equally likely to occur.

    More specifically, the probability of the factual case is ``0.5 - bias``,
    and the probability of each counterfactual case is ``(0.5 + bias) / num_actions``,
    where ``num_actions`` is the number of counterfactual actions for the sample site (usually 1).

    :param actions: A mapping from sample site names to interventions.
    :param bias: The scalar bias towards not intervening. Must be between -0.5 and 0.5.
    :param prefix: The prefix for naming the auxiliary discrete random variables.
    """

    actions: Mapping[str, Intervention[T]]
    prefix: str
    bias: float

    def __init__(
        self,
        actions: Mapping[str, Intervention[T]],
        *,
        prefix: str = "__witness_split_",
        bias: float = 0.0,
    ):
        assert -0.5 <= bias <= 0.5, "bias must be between -0.5 and 0.5"
        self.actions = actions
        self.bias = bias
        self.prefix = prefix
        super().__init__()

    def _pyro_post_sample(self, msg):
        try:
            action = self.actions[msg["name"]]
        except KeyError:
            return

        action = (action,) if not isinstance(action, tuple) else action
        num_actions = len(action) if isinstance(action, tuple) else 1
        weights = torch.tensor(
            [0.5 - self.bias] + ([(0.5 + self.bias) / num_actions] * num_actions),
            device=msg["value"].device,
        )
        case_dist = pyro.distributions.Categorical(probs=weights)
        case = pyro.sample(f"{self.prefix}{msg['name']}", case_dist)

        msg["value"] = preempt(
            msg["value"],
            action,
            case,
            event_dim=len(msg["fn"].event_shape),
            name=f"{self.prefix}{msg['name']}",
        )
