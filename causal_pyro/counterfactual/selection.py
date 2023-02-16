from typing import Any, Dict, Optional, TypeVar

import pyro
import pyro.infer.reparam
import torch

from causal_pyro.primitives import IndexSet, merge
from causal_pyro.counterfactual.internals import get_index_plates, indexset_as_mask

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
        return IndexSet(**{f.name: set(range(1, f.size)) for f in get_index_plates().values()})


class OnlyFactual(IndexSetMaskMessenger):
    """
    Effect handler to select only factual world.
    """

    @property
    def indices(self) -> IndexSet:
        return IndexSet(**{f.name: {0} for f in get_index_plates().values()})


class FactualConditioningReparam(pyro.infer.reparam.reparam.Reparam):
    def apply(self, msg):
        with OnlyFactual() as fw:
            fv = pyro.sample(msg["name"] + "_factual", msg["fn"], obs=msg["value"])
            fw_indices = fw.indices

        with OnlyCounterfactual() as cw:
            cv = pyro.sample(msg["name"] + "_counterfactual", msg["fn"])
            cw_indices = cw.indices

        event_dim = len(msg["fn"].event_shape)
        new_value = merge({fw_indices: fv, cw_indices: cv}, event_dim=event_dim)
        new_fn = pyro.distributions.Delta(new_value, event_dim=event_dim).mask(False)
        return {"fn": new_fn, "value": new_value, "is_observed": msg["is_observed"]}


class FactualConditioning(pyro.infer.reparam.strategies.Strategy):
    def configure(self, msg) -> Optional[FactualConditioningReparam]:
        if not msg["is_observed"] or pyro.poutine.util.site_is_subsample(msg) or \
                msg["name"].endswith("_factual"):
            return None
        return FactualConditioningReparam()
