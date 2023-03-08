from typing import Any, Dict

import pyro

from causal_pyro.counterfactual.internals import (
    get_index_plates,
    get_sample_msg_device,
    indexset_as_mask,
)
from causal_pyro.primitives import IndexSet


class IndexSetMaskMessenger(pyro.poutine.messenger.Messenger):
    """
    Effect handler to select a subset of worlds.
    """

    indices: IndexSet

    def _pyro_sample(self, msg: Dict[str, Any]) -> None:
        mask_device = get_sample_msg_device(msg["fn"], msg["value"])
        mask = indexset_as_mask(self.indices, device=mask_device)
        msg["mask"] = mask if msg["mask"] is None else msg["mask"] & mask


def factual_indices() -> IndexSet:
    return IndexSet(**{f.name: {0} for f in get_index_plates().values()})


def counterfactual_indices() -> IndexSet:
    return IndexSet(
        **{f.name: set(range(1, f.size)) for f in get_index_plates().values()}
    )


class SelectCounterfactual(IndexSetMaskMessenger):
    """
    Effect handler to select only counterfactual worlds.
    """

    @property
    def indices(self) -> IndexSet:
        return counterfactual_indices()


class SelectFactual(IndexSetMaskMessenger):
    """
    Effect handler to select only factual world.
    """

    @property
    def indices(self) -> IndexSet:
        return factual_indices()
