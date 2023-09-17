# RU: this is temporary file to play around
# will be removed later


import pytest

import contextlib
import itertools
import logging

import pyro
import pyro.distributions as dist
import pytest
import torch


from chirho.counterfactual.handlers import MultiWorldCounterfactual
from chirho.counterfactual.handlers.explanation import undo_split
from chirho.counterfactual.ops import split

from chirho.indexed.handlers import IndexPlatesMessenger
from chirho.indexed.internals import add_indices
from chirho.indexed.ops import (
    IndexSet,
    cond,
    gather,
    get_index_plates,
    indexset_as_mask,
    indices_of,
    scatter,
    union,
)

import logging
from typing import Iterable

import pyro
import pyro.distributions as dist
import pyro.infer
import pytest
import torch
from pyro.infer.autoguide import AutoMultivariateNormal

import chirho.interventional.handlers  # noqa: F401
from chirho.counterfactual.handlers import (  # TwinWorldCounterfactual,
    MultiWorldCounterfactual,
    SingleWorldCounterfactual,
    SingleWorldFactual,
    TwinWorldCounterfactual,
)
from chirho.counterfactual.handlers.counterfactual import BiasedPreemptions, Preemptions
from chirho.counterfactual.handlers.selection import SelectFactual
from chirho.counterfactual.ops import preempt, split
from chirho.indexed.ops import IndexSet, gather, indices_of, union
from chirho.interventional.handlers import do
from chirho.interventional.ops import intervene
from chirho.observational.handlers import condition
from chirho.observational.handlers.soft_conditioning import AutoSoftConditioning
from chirho.observational.ops import observe



 
#@pytest.mark.parametrize("cf_dim", [-2, -3, None])
#@pytest.mark.parametrize("event_shape", [(), (3,), (3, 2)])
#def test_cf_handler_preemptions(cf_dim, event_shape):

plate_size = 4
event_shape = ()

cf_dim = None

event_dim = len(event_shape)

#splits = {"x": torch.tensor(0.0)}

# w = pyro.sample(
#         "w", dist.Normal(0, 1).expand(event_shape).to_event(len(event_shape)))
# print(w)

shape = torch.Size([plate_size, *event_shape])
print(shape)
replace1 = torch.ones(shape)
print("replace", replace1, replace1.shape)

preemption_tensor = replace1 * 5 
print("preemption_tensor", preemption_tensor, preemption_tensor.shape)

# print("replace", replace1, replace1.shape)
# print("preempt", preempt, preempt.shape)
case = torch.randint(0, 2, size=shape)
print("case", case, case.shape)




#@do(actions=splits)
@pyro.plate("data", size=plate_size, dim=-1)
def model():
    w = pyro.sample(
        "w", dist.Normal(0, 1).expand(event_shape).to_event(len(event_shape)))
    print(w, w.shape)


    w = split(w, (replace1,), name="split1")
    print("split1", w)


    w = pyro.deterministic(
            "w_preempted",
            preempt(w, preemption_tensor, case, name="w_preempted")
    )
    print("w_preempted", w)

    w = pyro.deterministic(
            "w_undone", undo_split(antecedents=["w_preempted"])(w)
        )
    
    print("w_undone", w)


    



with MultiWorldCounterfactual() as mwc:
    with pyro.poutine.trace() as tr:
       model()

nd = tr.trace.nodes
    
#    .expand(event_shape).to_event(len(event_shape)

    # x = pyro.sample("x", dist.Normal(w, 1).to_event(len(event_shape)))
    # y = pyro.sample("y", dist.Normal(w + x, 1).to_event(len(event_shape)))
    # z = pyro.sample("z", dist.Normal(x + y, 1).to_event(len(event_shape)))
    # return dict(w=w, x=x, y=y, z=z)



# Z = intervene(
#             Z, torch.full(event_shape, x_cf_value - 1.0), event_dim=len(event_shape)
#         )

# preemption_handler = Preemptions(actions=preemptions, bias=0.1, prefix="__split_")

#with MultiWorldCounterfactual(cf_dim), preemption_handler:
#     tr = pyro.poutine.trace(model).get_trace()
#     assert all(f"__split_{k}" in tr.nodes for k in preemptions.keys())
#     assert indices_of(tr.nodes["w"]["value"], event_dim=event_dim) == IndexSet()
#     assert indices_of(tr.nodes["y"]["value"], event_dim=event_dim) == IndexSet(
#         x={0, 1}
#     )
#     assert indices_of(tr.nodes["z"]["value"], event_dim=event_dim) == IndexSet(
#         x={0, 1}
#     )

