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
        msg["kwargs"]["name"] = f"__split_{msg['name']}"
        case_dist = pyro.distributions.Categorical(torch.ones(len(acts) + 1))
        case = pyro.sample(msg["kwargs"]["name"], case_dist.mask(False), obs=case)
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
    """

    actions: Mapping[str, Intervention[T]]

    def __init__(self, actions: Mapping[str, Intervention[T]]):
        self.actions = actions
        super().__init__()

    def _pyro_post_sample(self, msg: Dict[str, Any]) -> None:
        try:
            action = self.actions[msg["name"]]
        except KeyError:
            return
        msg["value"] = preempt(
            msg["value"],
            (action,),
            None,
            event_dim=len(msg["fn"].event_shape),
            name=msg["name"],
        )
