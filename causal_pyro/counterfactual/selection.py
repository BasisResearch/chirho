from typing import Any, Dict, Optional

import pyro
import torch

from causal_pyro.counterfactual.internals import (
    get_index_plates,
    get_sample_msg_device,
    indexset_as_mask,
)
from causal_pyro.primitives import IndexSet


class DependentMaskMessenger(pyro.poutine.messenger.Messenger):
    """
    Abstract base class for effect handlers that select a subset of worlds.
    """

    def get_mask(
        self,
        dist: pyro.distributions.Distribution,
        value: Optional[torch.Tensor],
        device: torch.device = torch.device("cpu"),
    ) -> torch.Tensor:
        raise NotImplementedError

    def _pyro_sample(self, msg: Dict[str, Any]) -> None:
        device = get_sample_msg_device(msg["fn"], msg["value"])
        mask = self.get_mask(msg["fn"], msg["value"], device=device)
        msg["mask"] = mask if msg["mask"] is None else msg["mask"] & mask


def get_factual_indices() -> IndexSet:
    """
    Helpful operation used with :class:`MultiWorldCounterfactual`
    that returns an :class:`IndexSet` corresponding to the factual world,
    i.e. the world with index 0 for each index variable where no interventions
    have been performed.

    :return: IndexSet corresponding to the factual world.
    """
    return IndexSet(**{f.name: {0} for f in get_index_plates().values()})


class SelectCounterfactual(DependentMaskMessenger):
    """
    Effect handler to include only log-density terms from counterfactual worlds.
    This implementation piggybacks on Pyro's existing masking functionality,
    as used in :class:`pyro.poutine.mask.MaskMessenger` and elsewhere.

    Useful for transformations that require different behavior in the factual
    and counterfactual worlds, such as conditioning.

    .. note:: Semantically equivalent to applying the following at each sample site:

        pyro.poutine.mask(mask=~indexset_as_mask(get_factual_indices()))
    """

    def get_mask(
        self,
        dist: pyro.distributions.Distribution,
        value: Optional[torch.Tensor],
        device: torch.device = torch.device("cpu"),
    ) -> torch.Tensor:
        indices = get_factual_indices()
        return ~indexset_as_mask(indices, device=device)  # negate == complement


class SelectFactual(DependentMaskMessenger):
    """
    Effect handler to include only log-density terms from the factual world.
    This implementation piggybacks on Pyro's existing masking functionality,
    as used in :class:`pyro.poutine.mask.MaskMessenger` and elsewhere.

    Useful for transformations that require different behavior in the factual
    and counterfactual worlds, such as conditioning.

    .. note:: Semantically equivalent to applying the following at each sample site:

        pyro.poutine.mask(mask=indexset_as_mask(get_factual_indices()))
    """

    def get_mask(
        self,
        dist: pyro.distributions.Distribution,
        value: Optional[torch.Tensor] = None,
        device: torch.device = torch.device("cpu"),
    ) -> torch.Tensor:
        indices = get_factual_indices()
        return indexset_as_mask(indices, device=device)
