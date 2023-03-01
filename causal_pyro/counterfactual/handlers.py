from typing import Any, Dict

import pyro

from causal_pyro.counterfactual.internals import IndexPlatesMessenger
from causal_pyro.primitives import IndexSet, scatter


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
            msg["name"] = "__intervention__"
        if msg["name"] in self.plates:
            msg["name"] = f"{msg['name']}_{self.first_available_dim}"
        name = msg["name"]

        msg["value"] = scatter({
            IndexSet(**{name: {0}}): obs,
            IndexSet(**{name: {1}}): act,
        }, event_dim=event_dim)


class TwinWorldCounterfactual(IndexPlatesMessenger, BaseCounterfactual):
    def _pyro_post_intervene(self, msg):
        obs, act = msg["args"][0], msg["value"]
        event_dim = msg["kwargs"].setdefault("event_dim", 0)
        # disregard the name
        name = "__intervention__"

        msg["value"] = scatter({
            IndexSet(**{name: {0}}): obs,
            IndexSet(**{name: {1}}): act,
        }, event_dim=event_dim)
