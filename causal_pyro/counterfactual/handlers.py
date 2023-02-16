from typing import Any, Dict, List, Optional

import pyro

import causal_pyro.primitives
import causal_pyro.counterfactual.internals

from causal_pyro.primitives import IndexSet, join, merge
from causal_pyro.counterfactual.internals import IndexPlatesMessenger, add_indices


class BaseCounterfactual(pyro.poutine.messenger.Messenger):
    """
    Base class for counterfactual handlers.
    """

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
            msg["name"] = f"intervention_{self.first_available_dim}"
        name = msg["name"]

        obs_indices = IndexSet(**{name: {0}})
        act_indices = IndexSet(**{name: {1}})
        add_indices(join(obs_indices, act_indices))

        msg["value"] = merge({obs_indices: obs, act_indices: act}, event_dim=event_dim)


class TwinWorldCounterfactual(IndexPlatesMessenger, BaseCounterfactual):
    def _pyro_post_intervene(self, msg):
        obs, act = msg["args"][0], msg["value"]
        event_dim = msg["kwargs"].setdefault("event_dim", 0)
        if msg["name"] is None:
            msg["name"] = "intervention"  # repeat the name
        name = msg["name"]

        obs_indices = IndexSet(**{name: {0}})
        act_indices = IndexSet(**{name: {1}})
        add_indices(join(obs_indices, act_indices))

        msg["value"] = merge({obs_indices: obs, act_indices: act}, event_dim=event_dim)
