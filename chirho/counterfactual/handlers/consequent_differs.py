# RAFAL: this is a temporary file for `consequent_differs` extraction
# and testing
# Will be removed once a proper PR is created

import collections.abc
import contextlib
import functools
from typing import Callable, Iterable, Mapping, ParamSpec, TypeVar

import pyro
import pyro.distributions as dist
import torch


from chirho.counterfactual.handlers.counterfactual import (
    BiasedPreemptions,
    MultiWorldCounterfactual,
)
from chirho.counterfactual.handlers.selection import get_factual_indices
from chirho.indexed.ops import IndexSet, cond, gather, indices_of, scatter
from chirho.interventional.handlers import do
from chirho.interventional.ops import Intervention
from chirho.observational.handlers.condition import Factors

P = ParamSpec("P")
T = TypeVar("T")


def consequent_differs(
    antecedents: Iterable[str] = [], eps: float = -1e8, event_dim: int = 0
) -> Callable[[T], torch.Tensor]:
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
            not_eq = torch.all(not_eq, dim=-1, keepdim=False)
        return cond(eps, 0.0, not_eq, event_dim=event_dim)

    return _consequent_differs


def ff_conjunctive():
    u_match_dropped = pyro.sample("u_match_dropped", dist.Bernoulli(0.5))
    u_lightning = pyro.sample("u_lightning", dist.Bernoulli(0.5))

    match_dropped = pyro.deterministic("match_dropped", u_match_dropped, event_dim=0)
    lightning = pyro.deterministic("lightning", u_lightning, event_dim=0)

    match_dropped = torch.tensor(match_dropped)
    lightning = torch.tensor(lightning)

    forest_fire = pyro.deterministic(
        "forest_fire", torch.logical_and(match_dropped, lightning), event_dim=0
    ).float()

    return {
        "match_dropped": match_dropped,
        "lightning": lightning,
        "forest_fire": forest_fire,
    }


antecedent = {"match_dropped": 0.0}
observations = {"match_dropped": 1.0, "lightning": 1.0}

cd = consequent_differs(["match_dropped"])

with MultiWorldCounterfactual() as mwc:
    with do(actions=antecedent):
        with pyro.condition(data=observations):
            with pyro.poutine.trace() as tr:
                ff_conjunctive()


nd = tr.trace.nodes

print(nd["forest_fire"]["value"])

# with MultiWorldCounterfactual():
# #with do(actions=self.antecedents_dict):

#                 with pyro.plate("plate", self.sample_size):
#                     self.consequent = self.model()[self.outcome]
#                     self.intervened_consequent = gather(
#                         self.consequent,
#                         IndexSet(**{ant: {1} for ant in self.antecedents}),
#                     )
#                     self.observed_consequent = gather(
#                         self.consequent,
#                         IndexSet(**{ant: {0} for ant in self.antecedents}),
#                     )
#                     self.consequent_differs = (
#                         self.intervened_consequent != self.observed_consequent
#                     )
#                     pyro.factor(
#                         "consequent_differs",
#                         torch.where(
#                             self.consequent_differs,
#                             torch.tensor(0.0),
#                             torch.tensor(-1e8),
#                         ),
#                     )

# self.trace = trace.trace
