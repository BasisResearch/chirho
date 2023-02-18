from typing import Any, Dict, Optional, TypeVar

import pyro
import pyro.infer.reparam
import torch

from causal_pyro.counterfactual.internals import get_index_plates, indexset_as_mask
from causal_pyro.primitives import IndexSet, merge

T = TypeVar("T")


class IndexSetMaskMessenger(pyro.poutine.messenger.Messenger):
    """
    Effect handler to select a subset of worlds.
    """

    @property
    def indices(self) -> IndexSet:
        raise NotImplementedError

    def _pyro_sample(self, msg: Dict[str, Any]) -> None:
        mask = indexset_as_mask(self.indices)
        msg["mask"] = msg["mask"] & mask if msg["mask"] is not None else mask


class OnlySelected(IndexSetMaskMessenger):
    """
    Effect handler to select a subset of worlds.
    """

    def __init__(self, indices: IndexSet):
        self._indices = indices
        super().__init__()

    @property
    def indices(self) -> IndexSet:
        return self._indices


class OnlyCounterfactual(IndexSetMaskMessenger):
    """
    Effect handler to select only counterfactual worlds.
    """

    @property
    def indices(self) -> IndexSet:
        return IndexSet(
            **{f.name: set(range(1, f.size)) for f in get_index_plates().values()}
        )


class OnlyFactual(IndexSetMaskMessenger):
    """
    Effect handler to select only factual world.
    """

    @property
    def indices(self) -> IndexSet:
        return IndexSet(**{f.name: {0} for f in get_index_plates().values()})


class FactualConditioningReparam(pyro.infer.reparam.reparam.Reparam):
    @pyro.poutine.infer_config(config_fn=lambda msg: {"_cf_conditioned": True})
    def apply(self, msg):
        with OnlyFactual() as fw:
            fv = pyro.sample(msg["name"] + "_factual", msg["fn"], obs=msg["value"])

        with OnlyCounterfactual() as cw:
            cv = pyro.sample(msg["name"] + "_counterfactual", msg["fn"])

        event_dim = len(msg["fn"].event_shape)
        new_value = merge({fw.indices: fv, cw.indices: cv}, event_dim=event_dim)
        new_fn = pyro.distributions.Delta(new_value, event_dim=event_dim).mask(False)
        return {"fn": new_fn, "value": new_value, "is_observed": msg["is_observed"]}


class FactualConditioning(pyro.infer.reparam.strategies.Strategy):
    @staticmethod
    def _expand_msg_value(msg: dict) -> None:
        _custom_init = getattr(msg["value"], "_pyro_custom_init", False)
        msg["value"] = msg["value"].expand(
            torch.broadcast_shapes(
                msg["fn"].batch_shape + msg["fn"].event_shape, msg["value"].shape
            )
        )
        msg["value"]._pyro_custom_init = _custom_init

    def configure(self, msg) -> Optional[FactualConditioningReparam]:
        if (
            not msg["is_observed"]
            or pyro.poutine.util.site_is_subsample(msg)
            or msg["infer"].get("_cf_conditioned", False)  # don't apply recursively
        ):
            return None

        if msg["is_observed"] and msg["value"] is not None:
            # XXX slightly gross workaround that mutates the msg in place to avoid
            #   triggering overzealous validation logic in pyro.poutine.reparam
            #   that uses cheaper tensor shape and identity equality checks as
            #   a conservative proxy for an expensive tensor value equality check.
            #   (see https://github.com/pyro-ppl/pyro/blob/685c7adee65bbcdd6bd6c84c834a0a460f2224eb/pyro/poutine/reparam_messenger.py#L99)  # noqa: E501
            #   This workaround is correct because FactualConditioningReparam does not change
            #   the values of the observation, it just packs counterfactual values around it;
            #   the equality check being approximated by that logic would still pass.
            self._expand_msg_value(msg)

        return FactualConditioningReparam()
