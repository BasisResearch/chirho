# to be deleted after the new version of the code is tested


import collections.abc
import contextlib
import functools
from typing import Callable, Iterable, Mapping, ParamSpec, TypeVar

import pyro
import pyro.distributions as dist
import torch

from chirho.counterfactual.handlers.counterfactual import MultiWorldCounterfactual, Preemptions
from chirho.counterfactual.handlers.selection import get_factual_indices
from chirho.indexed.ops import IndexSet, cond, gather, indices_of, scatter
from chirho.counterfactual.handlers.explanation import consequent_differs, undo_split
from chirho.interventional.handlers import do
from chirho.interventional.ops import Intervention
from chirho.observational.handlers.condition import Factors, condition

P = ParamSpec("P")
T = TypeVar("T")


@contextlib.contextmanager
def PartOfCause(
    actions: Mapping[str, Intervention[T]],
    *,
    bias: float = 0.0,
    prefix: str = "__cause_split_",
):
    # TODO support event_dim != 0 propagation in factual_preemption
    preemptions = {
        antecedent: undo_split(antecedents=[antecedent])
        for antecedent in actions.keys()
    }

    with do(actions=actions):
        with Preemptions(actions=preemptions, bias=bias, prefix=prefix):
            with pyro.poutine.trace() as logging_tr:
                yield logging_tr.trace



#_________________________________________________________________
# Bayesian stones throwing model__________________________________
def stones_bayesian_model():
    prob_sally_throws = pyro.sample("prob_sally_throws", dist.Beta(1, 1))
    prob_bill_throws = pyro.sample("prob_bill_throws", dist.Beta(1, 1))
    prob_sally_hits = pyro.sample("prob_sally_hits", dist.Beta(1, 1))
    prob_bill_hits = pyro.sample("prob_bill_hits", dist.Beta(1, 1))
    prob_bottle_shatters_if_sally = pyro.sample(
        "prob_bottle_shatters_if_sally", dist.Beta(1, 1)
    )
    prob_bottle_shatters_if_bill = pyro.sample(
        "prob_bottle_shatters_if_bill", dist.Beta(1, 1)
    )

    sally_throws = pyro.sample("sally_throws", dist.Bernoulli(prob_sally_throws))
    bill_throws = pyro.sample("bill_throws", dist.Bernoulli(prob_bill_throws))

    new_shp = torch.where(sally_throws == 1, prob_sally_hits, 0.0)

    sally_hits = pyro.sample("sally_hits", dist.Bernoulli(new_shp))

    new_bhp = torch.where(
        (bill_throws.bool() & (~sally_hits.bool())) == 1,
        prob_bill_hits,
        torch.tensor(0.0),
    )

    bill_hits = pyro.sample("bill_hits", dist.Bernoulli(new_bhp))

    new_bsp = torch.where(
        bill_hits.bool() == 1,
        prob_bottle_shatters_if_bill,
        torch.where(
            sally_hits.bool() == 1,
            prob_bottle_shatters_if_sally,
            torch.tensor(0.0),
        ),
    )

    bottle_shatters = pyro.sample("bottle_shatters", dist.Bernoulli(new_bsp))

    return {
        "sally_throws": sally_throws,
        "bill_throws": bill_throws,
        "sally_hits": sally_hits,
        "bill_hits": bill_hits,
        "bottle_shatters": bottle_shatters,
    }
# model ends___________________________________________________________________
#______________________________________________________________________________

#def test_single_layer_stones():


observations = {
    "prob_sally_throws": 1.0,
    "prob_bill_throws": 1.0,
    "prob_sally_hits": 1.0,
    "prob_bill_hits": 1.0,
    "prob_bottle_shatters_if_sally": 1.0,
    "prob_bottle_shatters_if_bill": 1.0,
}

with MultiWorldCounterfactual() as mwc:
    with PartOfCause({"sally_throws": 0.0}, bias=0.0) as tr:
        with condition(data={k: torch.as_tensor(v) for k, v in observations.items()}):
            stones_bayesian_model()




print("run!")
