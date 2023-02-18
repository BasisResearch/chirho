from typing import Any, Dict

import pyro

from causal_pyro.counterfactual.internals import (
    IndexPlatesMessenger,
    add_indices,
    get_index_plates,
    indexset_as_mask,
)
from causal_pyro.primitives import IndexSet, indices_of, join, merge


class BaseCounterfactual(pyro.poutine.messenger.Messenger):
    """
    Base class for counterfactual handlers.
    """

    def _pyro_intervene(self, msg: Dict[str, Any]) -> None:
        msg["stop"] = True

    def _pyro_sample(self, msg: Dict[str, Any]) -> None:
        if (
            not msg["is_observed"]
            or msg["value"] is None
            or pyro.poutine.util.site_is_subsample(msg)
            or msg["infer"].get("_cf_conditioned", False)  # don't apply recursively
            or not indices_of(msg["fn"])
        ):
            return None

        with pyro.poutine.infer_config(config_fn=lambda msg: {"_cf_conditioned": True}):
            with SelectFactual() as fw:
                fv = pyro.sample(msg["name"] + "_factual", msg["fn"], obs=msg["value"])

            with SelectCounterfactual() as cw:
                cv = pyro.sample(msg["name"] + "_counterfactual", msg["fn"])

        event_dim = len(msg["fn"].event_shape)
        msg["value"] = merge({fw.indices: fv, cw.indices: cv}, event_dim=event_dim)
        msg["fn"] = pyro.distributions.Delta(msg["value"], event_dim=event_dim).mask(
            False
        )


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


class IndexSetMaskMessenger(pyro.poutine.messenger.Messenger):
    """
    Effect handler to select a subset of worlds.
    """

    @property
    def indices(self) -> IndexSet:
        raise NotImplementedError

    def _pyro_sample(self, msg: Dict[str, Any]) -> None:
        mask = indexset_as_mask(self.indices)
        msg["mask"] = mask if msg["mask"] is None else msg["mask"] & mask


class SelectCounterfactual(IndexSetMaskMessenger):
    """
    Effect handler to select only counterfactual worlds.
    """

    @property
    def indices(self) -> IndexSet:
        return IndexSet(
            **{f.name: set(range(1, f.size)) for f in get_index_plates().values()}
        )


class SelectFactual(IndexSetMaskMessenger):
    """
    Effect handler to select only factual world.
    """

    @property
    def indices(self) -> IndexSet:
        return IndexSet(**{f.name: {0} for f in get_index_plates().values()})
