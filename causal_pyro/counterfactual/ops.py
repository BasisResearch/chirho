from typing import Optional, TypeVar

import pyro

from .index_set import IndexSet, gather, indices_of, scatter
from .worlds import (
    IndexPlatesMessenger,
    add_indices,
    complement,
    get_full_index,
    indexset_as_mask,
)

T, I = TypeVar("T"), TypeVar("I")


class MultiWorldInterventions(IndexPlatesMessenger):
    def _pyro_post_intervene(self, msg):
        obs, act = msg["args"][0], msg["value"]
        event_dim = msg["kwargs"].setdefault("event_dim", 0)
        name = msg.setdefault("name", f"intervention_{self.first_available_dim}")

        obs_indices = IndexSet(**{name: {0}})
        act_indices = IndexSet.difference(
            indices_of(act, event_dim=event_dim), obs_indices
        )

        add_indices(IndexSet(**{name: set(range(0, len(act_indices)))}))

        obs = gather(obs, obs_indices, event_dim=event_dim)
        act = gather(act, act_indices, event_dim=event_dim)

        msg["value"] = scatter(obs, obs_indices, result=act, event_dim=event_dim)


class InWorlds(pyro.poutine.mask_messenger.MaskMessenger):
    prefix: Optional[str]

    def __init__(self, prefix: Optional[str] = None):
        if prefix is not None:
            self.prefix = prefix
        super().__init__()

    @property
    def world(self) -> IndexSet:
        raise NotImplementedError

    def _pyro_sample(self, msg):
        msg["mask"] = msg["mask"] & indexset_as_mask(self.world, event_dim=0)


class Factual(InWorlds):
    prefix: str = "factual"

    @property
    def world(self):
        return IndexSet(**{name: {0} for name in get_full_index()})


class Counterfactual(InWorlds):
    prefix: str = "counterfactual"

    @property
    def world(self):
        return complement(IndexSet(**{name: {0} for name in get_full_index()}))
