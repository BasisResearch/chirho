import contextlib
import functools
import itertools
import numbers
from typing import Any, Dict, List, Optional, Set, TypeVar, Union

import pyro
import torch

from .index_set import IndexSet, gather, indices_of, scatter
from .worlds import (add_indices, get_full_index, get_index_plates,
                     indexset_as_mask)

T = TypeVar("T")


class SelectWorldsMessenger(pyro.contrib.autoname.scoping.ScopeMessenger):
    """
    Effect handler to select a subset of worlds.
    """
    @property
    def worlds(self) -> IndexSet:
        raise NotImplementedError

    def _pyro_sample(self, msg: Dict[str, Any]) -> None:
        msg["mask"] = msg["mask"] & indexset_as_mask(self.worlds)
        super()._pyro_sample(msg)


class OnlySelected(SelectWorldsMessenger):
    """
    Effect handler to select a subset of worlds.
    """
    def __init__(self, worlds: IndexSet, prefix: Optional[str] = None):
        self._worlds = worlds
        super().__init__(prefix=prefix)

    @property
    def worlds(self) -> IndexSet:
        return self._worlds


class OnlyCounterfactual(SelectWorldsMessenger):
    """
    Effect handler to select only counterfactual worlds.
    """
    @property
    def worlds(self) -> IndexSet:
        return IndexSet(**{f.name: set(range(1, f.size)) for f in get_index_plates()})


class OnlyFactual(SelectWorldsMessenger):
    """
    Effect handler to select only factual world.
    """
    @property
    def worlds(self) -> IndexSet:
        return IndexSet(**{f.name: {0} for f in get_index_plates()})
