from typing import Generic, Hashable, Mapping, TypeVar

import pyro

from chirho.observational.internals import ObserveNameMessenger
from chirho.observational.ops import AtomicObservation, observe

T = TypeVar("T")


class ConditionMessenger(Generic[T], ObserveNameMessenger):
    """
    Condition on values in a probabilistic program.

    Can be used as a drop-in replacement for :func:`pyro.condition` that supports
    a richer set of observational data types and enables counterfactual inference.
    """

    def __init__(self, data: Mapping[Hashable, AtomicObservation[T]]):
        self.data = data
        super().__init__()

    def _pyro_sample(self, msg):
        if pyro.poutine.util.site_is_subsample(msg) or pyro.poutine.util.site_is_factor(
            msg
        ):
            return

        if msg["name"] not in self.data or msg["infer"].get("_do_not_observe", None):
            if (
                "_markov_scope" in msg["infer"]
                and getattr(self, "_current_site", None) is not None
            ):
                msg["infer"]["_markov_scope"].pop(self._current_site, None)
            return

        msg["stop"] = True
        msg["done"] = True

        # flags to guarantee commutativity of condition, intervene, trace
        msg["mask"] = False
        msg["is_observed"] = False
        msg["infer"]["is_auxiliary"] = True
        msg["infer"]["_do_not_trace"] = True
        msg["infer"]["_do_not_intervene"] = True
        msg["infer"]["_do_not_observe"] = True

        with pyro.poutine.infer_config(
            config_fn=lambda msg_: {
                "_do_not_observe": msg["name"] == msg_["name"]
                or msg_["infer"].get("_do_not_observe", False)
            }
        ):
            try:
                self._current_site = msg["name"]
                msg["value"] = observe(
                    msg["fn"], self.data[msg["name"]], name=msg["name"], **msg["kwargs"]
                )
            finally:
                self._current_site = None


condition = pyro.poutine.handlers._make_handler(ConditionMessenger)[1]
