# to be removed once a separate PR for the inclusion of
# `consequent_differs` is ready

from typing import Callable, Iterable, TypeVar

import torch
import pyro
import pyro.distributions as dist
import pytest

from chirho.counterfactual.ops import split, preempt
from chirho.indexed.ops import IndexSet, cond, gather, indices_of
from chirho.counterfactual.handlers.selection import get_factual_indices
from chirho.counterfactual.handlers import MultiWorldCounterfactual

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
            not_eq = torch.all(not_eq, dim=-1, keepdim=False)
        return cond(eps, 0.0, not_eq, event_dim=event_dim)

    return _consequent_differs




#@pytest.mark.parametrize("plate_size", [4, 50, 200])
#@pytest.mark.parametrize("event_shape", [(), (3,), (3, 2)])
#def test_consequent_differs():

plate_size = 4
event_shape = ()
joint_dims = torch.Size([plate_size, *event_shape])
case = torch.randint(0, 2, size=joint_dims)


@pyro.plate("data", size=plate_size, dim=-1)
def model_cd():
    w = pyro.sample("w", dist.Normal(0, .1).expand(event_shape).to_event(len(event_shape)))
    new_w = w.clone()
    new_w[1::2] = 10
    w = split(w, (new_w,), name="split")
    consequent = pyro.deterministic("consequent", w * .1)
    con_dif = pyro.deterministic("con_dif", consequent_differs(antecedents=["split"])(consequent))
    
    


with MultiWorldCounterfactual() as mwc:
    with pyro.poutine.trace() as tr:
        model_cd()

nd = tr.trace.nodes

with mwc: 
    int_con_dif = gather(nd["con_dif"]["value"], IndexSet(**{"split": {1}})).squeeze()

assert torch.all(int_con_dif[1::2] == 0.0)
assert torch.all(int_con_dif[0::2] == -1e8)

