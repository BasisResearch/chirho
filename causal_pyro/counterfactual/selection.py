from typing import Any, Dict, Optional, TypeVar

import pyro

from .index_set import IndexSet, merge
from .worlds import get_index_plates, indexset_as_mask

T = TypeVar("T")


class SelectWorldsMessenger(pyro.contrib.autoname.scoping.ScopeMessenger):
    """
    Effect handler to select a subset of worlds.
    """
    @property
    def indices(self) -> IndexSet:
        raise NotImplementedError

    def _pyro_sample(self, msg: Dict[str, Any]) -> None:
        msg["mask"] = msg["mask"] & indexset_as_mask(self.indices)
        super()._pyro_sample(msg)


class OnlySelected(SelectWorldsMessenger):
    """
    Effect handler to select a subset of worlds.
    """
    def __init__(self, worlds: IndexSet, prefix: Optional[str] = None):
        self._worlds = worlds
        super().__init__(prefix=prefix)

    @property
    def indices(self) -> IndexSet:
        return self._worlds


class OnlyCounterfactual(SelectWorldsMessenger):
    """
    Effect handler to select only counterfactual worlds.
    """
    @property
    def indices(self) -> IndexSet:
        return IndexSet(**{f.name: set(range(1, f.size)) for f in get_index_plates()})


class OnlyFactual(SelectWorldsMessenger):
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

        with OnlyFactual(prefix="factual") as fw:
            # TODO prevent unbounded recursion here
            fv = pyro.sample(msg["name"], msg["fn"], obs=msg["value"])

        with OnlyCounterfactual(prefix="counterfactual") as cw:
            cv = pyro.sample(msg["name"], msg["fn"])

        msg["value"] = merge({fw: fv, cw: cv}, event_dim=len(msg["fn"].event_shape))

        # emulate a deterministic statement
        msg["fn"] = pyro.distributions.Delta(
            msg["value"], event_dim=len(msg["fn"].event_shape)
        ).mask(False)
