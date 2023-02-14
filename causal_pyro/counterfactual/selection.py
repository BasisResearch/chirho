from typing import Any, Dict, Optional, TypeVar

import pyro

from .index_set import IndexSet, merge
from .worlds import get_index_plates, indexset_as_mask

T = TypeVar("T")


class IndexSetMaskMessenger(pyro.poutine.messenger.Messenger):
    """
    Effect handler to select a subset of worlds.
    """

    @property
    def indices(self) -> IndexSet:
        raise NotImplementedError

    def _pyro_sample(self, msg: Dict[str, Any]) -> None:
        msg["mask"] = msg["mask"] & indexset_as_mask(self.indices)


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
        return IndexSet(**{f.name: set(range(1, f.size)) for f in get_index_plates()})


class OnlyFactual(IndexSetMaskMessenger):
    """
    Effect handler to select only factual world.
    """

    @property
    def indices(self) -> IndexSet:
        return IndexSet(**{f.name: {0} for f in get_index_plates()})


class OnlyFactualConditioningReparam(pyro.infer.reparam.reparam.Reparam):
    def apply(self, msg):
        if not msg["is_observed"] or pyro.poutine.util.site_is_subsample(msg):
            return

        with OnlyFactual() as fw:
            # TODO prevent unbounded recursion here
            fv = pyro.sample(msg["name"], msg["fn"], obs=msg["value"])

        with OnlyCounterfactual() as cw:
            cv = pyro.sample(msg["name"], msg["fn"])

        event_dim = len(msg["fn"].event_shape)
        msg["value"] = merge({fw.indices: fv, cw.indices: cv}, event_dim=event_dim)

        # emulate a deterministic statement
        msg["fn"] = pyro.distributions.Delta(msg["value"], event_dim=event_dim).mask(False)


class OnlyFactualConditioning(pyro.infer.reparam.strategies.Strategy):
    def configure(self, msg) -> Optional[OnlyFactualConditioningReparam]:
        if not msg["is_observed"]:
            return None
        return OnlyFactualConditioningReparam()
