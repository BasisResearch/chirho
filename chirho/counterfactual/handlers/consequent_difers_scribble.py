# to be removed once a separate PR for the inclusion of
# `consequent_differs` is ready

from typing import Callable, Iterable, TypeVar

import torch
import pyro
from pyro.distributions import Bernoulli

from chirho.indexed.ops import IndexSet, cond, gather
from chirho.counterfactual.handlers.selection import get_factual_indices

T = TypeVar("T")


def consequent_differs(antecedents: Iterable[str] = [], eps: float = -1e8, event_dim: int = 0) -> Callable[[T], torch.Tensor]:

    def _consequent_differs(consequent: T) -> torch.Tensor:
        indices = IndexSet(
            **{
                name: ind
                for name, ind in get_factual_indices().items()
                if name in antecedents
            }
        )
        not_eq = consequent != gather(consequent, indices, event_dim=event_dim)
        for _ in range(event_dim):
            not_eq = torch.all(
                not_eq, dim=-1, keepdim=False)
        return cond(eps, 0.0, not_eq, event_dim=event_dim)

    return _consequent_differs


# testing setup
# Simple model based on the bottle shattering example______
sally_hits_pt = torch.tensor([1.0])
bill_hits_cpt = torch.tensor([1.0, 0.0])
bottle_shatters_cpt = torch.tensor([[0.0, 1.0], [1.0, 1.0]])

probs = (sally_hits_pt, bill_hits_cpt, bottle_shatters_cpt)


def bottle_bn(sally_hits_pt, bill_hits_cpt, bottle_shatters_cpt):
    sh = pyro.sample("sh", Bernoulli(sally_hits_pt)).long()
    bh = pyro.sample("bh", Bernoulli(bill_hits_cpt[sh])).long()
    bs = pyro.sample("bs", Bernoulli(bottle_shatters_cpt[sh, bh])).long()

    # bh = bh.float()
    # bh = bh.float()
    # bs = bs.float()

    return {"sh": sh, "bh": bh, "bs": bs}


def bottle_bn_complete():
    return bottle_bn(*probs)


