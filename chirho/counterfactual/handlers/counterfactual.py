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
        obs, acts, case = msg["args"]
        if msg["kwargs"].get("name", None) is None:
            msg["kwargs"]["name"] = msg["name"]

        if case is not None:
            return

        case_dist = pyro.distributions.Categorical(torch.ones(len(acts) + 1))
        case = pyro.sample(msg["name"], case_dist.mask(False), obs=case)
        msg["args"] = (obs, acts, case)


class SingleWorldCounterfactual(BaseCounterfactualMessenger):
    """
    Trivial counterfactual handler that returns the intervened value.
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
    """

    @staticmethod
    def _pyro_split(msg: Dict[str, Any]) -> None:
        obs, _ = msg["args"]
        msg["value"] = obs
        msg["done"] = True
        msg["stop"] = True


class MultiWorldCounterfactual(IndexPlatesMessenger, BaseCounterfactualMessenger):
    default_name: str = "intervened"

    @classmethod
    def _pyro_split(cls, msg: Dict[str, Any]) -> None:
        name = msg["name"] if msg["name"] is not None else cls.default_name
        index_plates = get_index_plates()
        if name in index_plates:
            name = f"{name}__dup_{len(index_plates)}"
        msg["kwargs"]["name"] = msg["name"] = name


class TwinWorldCounterfactual(IndexPlatesMessenger, BaseCounterfactualMessenger):
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

    .. note:: This handler does not allow the direct specification of the ``case`` argument
        to :func:`~chirho.counterfactual.ops.preempt` and therefore cannot be used alone.
        Instead, the ``case`` argument to :func:`preempt` is assumed to be set separately
        by :class:`~chirho.counterfactual.handlers.counterfactual.BaseCounterfactualMessenger`
        or one of its subclasses, typically from an auxiliary discrete random variable.

    :param actions: A mapping from sample site names to interventions.
    :param prefix: Prefix usable for naming any auxiliary random variables.
    """

    actions: Mapping[str, Intervention[T]]
    prefix: str

    def __init__(
        self, actions: Mapping[str, Intervention[T]], *, prefix: str = "__split_"
    ):
        self.actions = actions
        self.prefix = prefix
        super().__init__()

    def _pyro_post_sample(self, msg):
        try:
            action = self.actions[msg["name"]]
        except KeyError:
            return
        msg["value"] = preempt(
            msg["value"],
            (action,) if not isinstance(action, tuple) else action,
            None,
            event_dim=len(msg["fn"].event_shape),
            name=f"{self.prefix}{msg['name']}",
        )


class BiasedPreemptions(Generic[T], pyro.poutine.messenger.Messenger):
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
    :param bias: The scalar bias towards the factual case. Must be between -0.5 and 0.5.
    :param prefix: The prefix for naming the auxiliary discrete random variables.
    """

    actions: Mapping[str, Intervention[T]]
    bias: float
    prefix: str

    def __init__(
        self,
        actions: Mapping[str, Intervention[T]],
        *,
        bias: float = 0.0,
        prefix: str = "__witness_split_",
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
