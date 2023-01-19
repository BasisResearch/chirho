from typing import Any, Dict, Optional, TypeVar

import pyro

from .index_set import IndexSet
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
