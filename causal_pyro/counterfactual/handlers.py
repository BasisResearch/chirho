from typing import Any, Dict, Optional, Union

import pyro

from causal_pyro.counterfactual.conditioning import (
    AmbiguousConditioningReparam,
    AmbiguousConditioningStrategy,
    AutoFactualConditioning,
)
from causal_pyro.counterfactual.internals import IndexPlatesMessenger, add_indices
from causal_pyro.primitives import IndexSet, join, merge


CondStrategy = Union[
    Dict[str, AmbiguousConditioningReparam],
    AmbiguousConditioningStrategy
]


class BaseCounterfactual(pyro.poutine.reparam_messenger.ReparamMessenger):
    """
    Base class for counterfactual handlers.
    """
    def __init__(self, config: Optional[CondStrategy] = None):
        if config is None:
            config = AutoFactualConditioning()
        super().__init__(config=config)

    def _pyro_intervene(self, msg: Dict[str, Any]) -> None:
        msg["stop"] = True


class Factual(BaseCounterfactual):
    """
    Trivial counterfactual handler that returns the observed value.
    """

    def _pyro_post_intervene(self, msg: Dict[str, Any]) -> None:
        obs, _ = msg["args"]
        msg["value"] = obs


class MultiWorldCounterfactual(IndexPlatesMessenger, BaseCounterfactual):
    def _pyro_post_intervene(self, msg):
        obs, act = msg["args"][0], msg["value"]
        event_dim = msg["kwargs"].setdefault("event_dim", 0)
        if msg["name"] is None:
            msg["name"] = "__intervention__"
        if msg["name"] in self.plates:
            msg["name"] = f"{msg['name']}_{self.first_available_dim}"
        name = msg["name"]

        obs_indices = IndexSet(**{name: {0}})
        act_indices = IndexSet(**{name: {1}})
        add_indices(join(obs_indices, act_indices))

        msg["value"] = merge({obs_indices: obs, act_indices: act}, event_dim=event_dim)


class TwinWorldCounterfactual(IndexPlatesMessenger, BaseCounterfactual):
    def _pyro_post_intervene(self, msg):
        obs, act = msg["args"][0], msg["value"]
        event_dim = msg["kwargs"].setdefault("event_dim", 0)
        # disregard the name
        name = "__intervention__"

        obs_indices = IndexSet(**{name: {0}})
        act_indices = IndexSet(**{name: {1}})
        add_indices(join(obs_indices, act_indices))

        msg["value"] = merge({obs_indices: obs, act_indices: act}, event_dim=event_dim)
