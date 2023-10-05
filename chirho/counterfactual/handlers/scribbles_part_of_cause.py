# to be deleted after the new version of the code is tested


import contextlib
from typing import Mapping, ParamSpec, TypeVar

import pyro
import pyro.distributions as dist
import torch

from chirho.counterfactual.handlers.counterfactual import (
    MultiWorldCounterfactual,
    Preemptions,
)
from chirho.counterfactual.handlers.explanation import undo_split

# from chirho.counterfactual.handlers.selection import get_factual_indices
from chirho.indexed.ops import IndexSet, gather, indices_of
from chirho.interventional.handlers import do
from chirho.interventional.ops import Intervention
from chirho.observational.handlers.condition import condition

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
                yield logging_tr.trace.nodes


# _________________________________________________________________
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
# ______________________________________________________________________________


def test_single_layer_stones():
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
            with condition(
                data={k: torch.as_tensor(v) for k, v in observations.items()}
            ):
                stones_bayesian_model()

    tr = tr.nodes

    with mwc:
        preempt_sally_throws = gather(
            tr["__cause_split_sally_throws"]["value"],
            IndexSet(**{"sally_throws": {0}}),
            event_dim=0,
        )

        int_sally_hits = gather(
            tr["sally_hits"]["value"], IndexSet(**{"sally_throws": {1}}), event_dim=0
        )
        obs_bill_hits = gather(
            tr["bill_hits"]["value"], IndexSet(**{"sally_throws": {0}}), event_dim=0
        )
        int_bill_hits = gather(
            tr["bill_hits"]["value"], IndexSet(**{"sally_throws": {1}}), event_dim=0
        )
        int_bottle_shatters = gather(
            tr["bottle_shatters"]["value"],
            IndexSet(**{"sally_throws": {1}}),
            event_dim=0,
        )

    outcome = {
        "preempt_sally_throws": preempt_sally_throws.item(),
        "int_sally_hits": int_sally_hits.item(),
        "obs_bill_hits": obs_bill_hits.item(),
        "int_bill_hits": int_bill_hits.item(),
        "intervened_bottle_shatters": int_bottle_shatters.item(),
    }

    assert list(outcome.values()) == [0, 0.0, 0.0, 1.0, 1.0] or list(
        outcome.values()
    ) == [1, 1.0, 0.0, 0.0, 1.0]


#_______________________________________________________________________________
# second test___________________________________________________________________

#@pytest.mark.parametrize("antecedents", [None, ("sally_throws",), ("bill_hits",)])
#def test_two_layer_stones(antecedents):

antecedents = ("sally_throws",)

observations = {
    "prob_sally_throws": 1.0,
    "prob_bill_throws": 1.0,
    "prob_sally_hits": 1.0,
    "prob_bill_hits": 1.0,
    "prob_bottle_shatters_if_sally": 1.0,
    "prob_bottle_shatters_if_bill": 1.0,
}

evaluated_node_counterfactual = {"sally_throws": 0.0}
witness_preemptions = {"bill_hits": undo_split(antecedents=antecedents)}

pinned_preemption_variables = {
    "preempt_sally_throws": torch.tensor(0),
    "witness_preempt_bill_hits": torch.tensor(1),
}

part_of_cause_handler = PartOfCause(evaluated_node_counterfactual, prefix="preempt_")

preemptions_handler = Preemptions(actions=witness_preemptions, prefix="witness_preempt_")

preemption_conditioning = condition(data=pinned_preemption_variables)

observations_conditioning = condition(data={k: torch.as_tensor(v) for k, v in observations.items()})

#part_of_cause_handler as tr, preemptions_handler, preemption_conditioning, observations_conditioning:


with MultiWorldCounterfactual() as mwc: 
    with do(actions=evaluated_node_counterfactual):
        with preemption_conditioning, preemptions_handler:
            with observations_conditioning:
                with pyro.poutine.trace() as tr: 
                    stones_bayesian_model()

tr = tr.trace.nodes

print(
"st",     tr["sally_throws"]["value"],
"sh",     tr["sally_hits"]["value"],

"bt",     tr["bill_throws"]["value"],
"bh",     tr["bill_hits"]["value"],

"bs",     tr["bottle_shatters"]["value"],
#    tr["preempt_sally_throws"]["value"],
)

with mwc:
    print (indices_of(tr["bill_hits"]["value"], event_dim=0))





# with mwc:
#     preempt_sally_throws = tr["preempt_sally_throws"]["value"]
    
#     int_sally_hits = gather(tr["sally_hits"]["value"], IndexSet(**{"sally_throws": {1}}), event_dim=0,)

#     preempt_bill_hits = tr["witness_preempt_bill_hits"]["value"]

#     obs_bill_hits = gather( tr["bill_hits"]["value"], IndexSet(**{"sally_throws": {0}}), event_dim=0, )

#     int_bill_hits = gather(tr["bill_hits"]["value"], IndexSet(**{"sally_throws": {1}}), event_dim=0,)

#     int_bottle_shatters = gather(tr["bottle_shatters"]["value"],IndexSet(**{"sally_throws": {1}}),event_dim=0,)

# outcome2 = {
#     "preempt_sally_throws": preempt_sally_throws.item(),
#     "int_sally_hits": int_sally_hits.item(),
#     "witness_preempt_bill_hits": preempt_bill_hits.item(),
#     "obs_bill_hits": obs_bill_hits.item(),
#     "int_bill_hits": int_bill_hits.item(),
#     "intervened_bottle_shatters": int_bottle_shatters.item(),
# }

# print(outcome2)

    # if antecedents == ("bill_hits",):
    #     with pytest.raises(AssertionError):
    #         assert outcome["int_bill_hits"] == outcome["obs_bill_hits"]
    # else:
    #     assert outcome["int_bill_hits"] == outcome["obs_bill_hits"]

#_______________________________________________________________________________