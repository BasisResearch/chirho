from __future__ import annotations

from typing import Callable, Generic, Hashable, Mapping, TypeVar, Union

import pyro
import torch

from chirho.observational.internals import ObserveNameMessenger
from chirho.observational.ops import AtomicObservation, observe

T = TypeVar("T")
R = Union[float, torch.Tensor]


class Factors(Generic[T], pyro.poutine.messenger.Messenger):
    """
    Effect handler that adds new log-factors to the unnormalized
    joint log-density of a probabilistic program.

    After a :func:`pyro.sample` site whose name appears in ``factors``,
    this handler inserts a new :func:`pyro.factor` site
    whose name is prefixed with the string ``prefix``
    and whose log-weight is the result of applying the corresponding function
    to the value of the sample site. ::

        >>> with Factors(factors={"x": lambda x: -(x - 1) ** 2}, prefix="__factor_"):
        ...   with pyro.poutine.trace() as tr:
        ...     x = pyro.sample("x", dist.Normal(0, 1))
        ... tr.trace.compute_log_prob()
        >>> assert {"x", "__factor_x"} <= set(tr.trace.nodes.keys())
        >>> assert torch.all(tr.trace.nodes["x"]["log_prob"] == -(x - 1) ** 2)

    :param factors: A mapping from sample site names to log-factor functions.
    :param prefix: The prefix to use for the names of the factor sites.
    """

    factors: Mapping[str, Callable[[T], R]]
    prefix: str

    def __init__(
        self,
        factors: Mapping[str, Callable[[T], R]],
        *,
        prefix: str = "__factor_",
    ):
        self.factors = factors
        self.prefix = prefix
        super().__init__()

    def _pyro_post_sample(self, msg: dict) -> None:
        try:
            factor = self.factors[msg["name"]]
        except KeyError:
            return

        pyro.factor(f"{self.prefix}{msg['name']}", factor(msg["value"]))


class Observations(Generic[T], ObserveNameMessenger):
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


condition = pyro.poutine.handlers._make_handler(Observations)[1]
