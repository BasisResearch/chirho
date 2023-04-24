from typing import Any, Dict, Optional, TypeVar

import pyro

from causal_pyro.counterfactual.handlers.ambiguity import (
    AmbiguousConditioningReparamMessenger,
    AutoFactualConditioning,
    CondStrategy,
)
from causal_pyro.counterfactual.ops import gen_intervene_name, split
from causal_pyro.indexed.handlers import IndexPlatesMessenger
from causal_pyro.indexed.ops import IndexSet, get_index_plates, scatter
from causal_pyro.interventional.ops import intervene

T = TypeVar("T")


class BaseCounterfactual(AmbiguousConditioningReparamMessenger):
    """
    Base class for counterfactual handlers.
    """

    default_name: str = "intervened"

    def __init__(self, config: Optional[CondStrategy] = None):
        if config is None:
            config = AutoFactualConditioning()
        super().__init__(config=config)

    @staticmethod
    def _pyro_get_index_plates(msg: Dict[str, Any]) -> None:
        msg["stop"], msg["done"] = True, True
        msg["value"] = {}

    @staticmethod
    def _pyro_intervene(msg: Dict[str, Any]) -> None:
        msg["stop"] = True
        if msg["args"][1] is not None:
            obs, acts = msg["args"][0], msg["args"][1]
            acts = acts(obs) if callable(acts) else acts
            acts = (acts,) if not isinstance(acts, tuple) else acts
            msg["value"] = split(obs, acts, name=msg["name"], **msg["kwargs"])
            msg["done"] = True

    @classmethod
    def _pyro_gen_intervene_name(cls, msg: Dict[str, Any]) -> None:
        if not msg["done"]:
            (name,) = msg["args"]
            msg["value"] = name if name is not None else cls.default_name
            msg["done"] = True

    @staticmethod
    @pyro.poutine.block(hide_types=["intervene"])
    def _pyro_split(msg: Dict[str, Any]) -> None:
        if msg["done"]:
            return

        obs, acts = msg["args"]
        name = gen_intervene_name(msg["name"])

        act_values: Dict[IndexSet, T] = {IndexSet(**{name: {0}}): obs}
        for i, act in enumerate(acts):
            act_values[IndexSet(**{name: {i + 1}})] = intervene(
                obs, act, **msg["kwargs"]
            )

        msg["value"] = scatter(act_values, event_dim=msg["kwargs"].get("event_dim", 0))
        msg["done"] = True


class SingleWorldCounterfactual(BaseCounterfactual):
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


class SingleWorldFactual(BaseCounterfactual):
    """
    Trivial counterfactual handler that returns the observed value.
    """

    @staticmethod
    def _pyro_split(msg: Dict[str, Any]) -> None:
        obs, _ = msg["args"]
        msg["value"] = obs
        msg["done"] = True
        msg["stop"] = True


class MultiWorldCounterfactual(IndexPlatesMessenger, BaseCounterfactual):
    @classmethod
    def _pyro_gen_intervene_name(cls, msg: Dict[str, Any]) -> None:
        (name,) = msg["args"]
        if name is None:
            name = cls.default_name
        index_plates = get_index_plates()
        if name in index_plates:
            name = f"{name}_{len(index_plates)}"
        msg["value"] = name
        msg["done"] = True


class TwinWorldCounterfactual(IndexPlatesMessenger, BaseCounterfactual):
    @classmethod
    def _pyro_gen_intervene_name(cls, msg: Dict[str, Any]) -> None:
        msg["value"] = cls.default_name
        msg["done"] = True
