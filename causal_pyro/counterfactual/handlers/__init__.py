from typing import Any, Dict, Optional, TypeVar

import pyro
import torch

from causal_pyro.counterfactual.handlers.ambiguity import (
    AmbiguousConditioningReparamMessenger,
    AutoFactualConditioning,
    CondStrategy,
)
from causal_pyro.counterfactual.ops import split
from causal_pyro.indexed.handlers import IndexPlatesMessenger
from causal_pyro.indexed.ops import get_index_plates
from causal_pyro.interventional.ops import intervene

T = TypeVar("T")


class BaseCounterfactualMessenger(AmbiguousConditioningReparamMessenger):
    """
    Base class for counterfactual handlers.
    """

    def __init__(self, config: Optional[CondStrategy] = None):
        if config is None:
            config = AutoFactualConditioning()
        super().__init__(config=config)

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
        case_dist = pyro.distributions.Categorical(torch.ones(len(acts) + 1)).mask(False)
        case = pyro.sample(f"__split_{msg['name']}", case_dist, obs=case)
        msg["args"] = (obs, acts, case)


class SingleWorldCounterfactual(BaseCounterfactualMessenger):
    """
    Trivial counterfactual handler that returns the intervened value.
    """

    @staticmethod
    @pyro.poutine.block(hide_types=["intervene"])
    def _pyro_split(msg: Dict[str, Any]) -> None:
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
            name = f"{name}_{len(index_plates)}"
        msg["kwargs"]["name"] = msg["name"] = name


class TwinWorldCounterfactual(IndexPlatesMessenger, BaseCounterfactualMessenger):
    default_name: str = "intervened"

    @classmethod
    def _pyro_split(cls, msg: Dict[str, Any]) -> None:
        msg["kwargs"]["name"] = msg["name"] = cls.default_name
