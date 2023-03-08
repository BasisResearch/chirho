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
    Abstract base clas for effect handlers that select a subset of worlds.
    """

    @property
    def indices(self) -> IndexSet:
        raise NotImplementedError

    def _pyro_sample(self, msg: Dict[str, Any]) -> None:
        mask_device = get_sample_msg_device(msg["fn"], msg["value"])
        mask = indexset_as_mask(self.indices, device=mask_device)
        msg["mask"] = mask if msg["mask"] is None else msg["mask"] & mask


def factual_indices() -> IndexSet:
    """
    Helpful operation used with :class:`MultiWorldCounterfactual`
    that returns an :class:`IndexSet` corresponding to the factual world,
    i.e. the world with index 0 for each index variable where no interventions
    have been performed.

    :return: IndexSet corresponding to the factual world.
    """
    return IndexSet(**{f.name: {0} for f in get_index_plates().values()})


def counterfactual_indices() -> IndexSet:
    """
    Helpful operation used with :class:`MultiWorldCounterfactual`
    that returns an :class:`IndexSet` corresponding to the counterfactual worlds,
    i.e. the worlds with indices 1, ..., n for each index variable where no
    interventions have been performed.

    :return: IndexSet corresponding to the counterfactual worlds.
    """
    return IndexSet(
        **{f.name: set(range(1, f.size)) for f in get_index_plates().values()}
    )


class SelectCounterfactual(IndexSetMaskMessenger):
    """
    Effect handler to include only log-density terms from counterfactual worlds.
    This implementation piggybacks on Pyro's existing masking functionality,
    as used in :class:`pyro.poutine.mask.MaskMessenger` and elsewhere.

    Useful for transformations that require different behavior in the factual
    and counterfactual worlds, such as conditioning.

    .. note:: Semantically equivalent to

        pyro.poutine.mask(mask=indexset_as_mask(counterfactual_indices()))
    """

    @property
    def indices(self) -> IndexSet:
        return counterfactual_indices()


class SelectFactual(IndexSetMaskMessenger):
    """
    Effect handler to include only log-density terms from the factual world.
    This implementation piggybacks on Pyro's existing masking functionality,
    as used in :class:`pyro.poutine.mask.MaskMessenger` and elsewhere.

    Useful for transformations that require different behavior in the factual
    and counterfactual worlds, such as conditioning.

    .. note:: Semantically equivalent to

        pyro.poutine.mask(mask=indexset_as_mask(factual_indices()))
    """

    @property
    def indices(self) -> IndexSet:
        return factual_indices()
