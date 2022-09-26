from typing import Callable, Optional, Union, TypeVar, Dict, Any

import pyro


class BaseCounterfactual(pyro.poutine.messenger.Messenger):
    """
    Base class for counterfactual handlers.
    """
    def _pyro_sample(self, msg: Dict[str, Any]) -> None:
        pass

    def _pyro_intervene(self, msg: Dict[str, Any]) -> None:
        pass


class Factual(BaseCounterfactual):
    """
    Trivial counterfactual handler that returns the observed value.
    """
    def _pyro_sample(self, msg: Dict[str, Any]) -> None:
        if not msg["done"]:
            obs, _ = msg["args"]
            msg["value"] = obs
            msg["done"] = True