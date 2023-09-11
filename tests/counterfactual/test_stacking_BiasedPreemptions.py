# WISH: there should be a principled way of representing both witness preemption on an antecedent-intervened variable,
# and antecedent intervention on a witness preempted variable. This is meant to facilitate the combined search for
# actual causes with minimal (witness set size + antecedent set size) if we plan to implement the original definition of
# responsibility in terms of log probs in the future.

import torch
import pyro
import pyro.distributions as dist

from chirho.counterfactual.handlers import (
    MultiWorldCounterfactual,
    TwinWorldCounterfactual,
)

from chirho.counterfactual.handlers.counterfactual import (
    BiasedPreemptions,
    MultiWorldCounterfactual,
)

from chirho.counterfactual.handlers.selection import get_factual_indices
from chirho.indexed.ops import IndexSet, cond, gather, indices_of, scatter
from chirho.interventional.handlers import do
from chirho.interventional.ops import Intervention
from chirho.observational.handlers import condition

# from chirho.observational.handlers.condition import Factors
from chirho.observational.ops import Observation


from typing import (
    Callable,
    Iterable,
    Mapping,
    Optional,
    ParamSpec,
    TypeVar,
    List,
)


T = TypeVar("T")


# factual preemption


def factual_preemption(
    antecedents: Optional[Iterable[str]] = None, event_dim: int = 0
) -> Callable[[T], T]:
    def _preemption_with_factual(value: T) -> T:
        if antecedents is None:
            antecedents_ = list(indices_of(value, event_dim=event_dim).keys())
        else:
            antecedents_ = [
                a for a in antecedents if a in indices_of(value, event_dim=event_dim)
            ]

        factual_value = gather(
            value,
            IndexSet(**{antecedent: {0} for antecedent in antecedents_}),
            event_dim=event_dim,
        )

        return scatter(
            {
                IndexSet(
                    **{antecedent: {0} for antecedent in antecedents_}
                ): factual_value,
                IndexSet(
                    **{antecedent: {1} for antecedent in antecedents_}
                ): factual_value,
            },
            event_dim=event_dim,
        )

    return _preemption_with_factual


# Model___________________________


# Bayesian version
@pyro.infer.config_enumerate
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


observations = {
    "prob_sally_throws": 1.0,
    "prob_bill_throws": 1.0,
    "prob_sally_hits": 1.0,
    "prob_bill_hits": 1.0,
    "prob_bottle_shatters_if_sally": 1.0,
    "prob_bottle_shatters_if_bill": 1.0,
}


stones_bayesian_model = condition(stones_bayesian_model, observations)


# Seting up interventions and preemptions________________________


# we want to consider the impact of sally throwing the stone
# while considering bill hitting as a possible antecedent
evaluated_node_counterfactual = {"sally_throws": 0.0}
antecedent_candidate_counterfactual = {"bill_hits": 1.0}


# witness preemptions contain witnesses alone as antecedents
witness_candidates = ["bill_hits"]
witness_preemptions = {
    candidate: factual_preemption(antecedents=[candidate])
    for candidate in witness_candidates
}


# evaluated node preemptions and antecedent preemptions
# contain all intervention candidates as antecedents
evaluated_node_preemptions = {
    candidate: factual_preemption(antecedents=[candidate])
    for candidate in ["sally_throws"]
    # including all intervention candidates fails too
    # for candidate in ["sally_throws", "bill_hits"]
}


antecedent_preemptions = {
    candidate: factual_preemption(antecedents=[candidate])
    for candidate in ["bill_hits"]
    # including all intervention candidates fails too
    # for candidate in ["sally_throws", "bill_hits"]
}


evaluated_node_preemption_handler = BiasedPreemptions(
    actions=evaluated_node_preemptions, bias=0.2, prefix="__evaluated_"
)

antecedent_node_preemption_handler = BiasedPreemptions(
    actions=antecedent_preemptions, bias=0.15, prefix="__antecedent_"
)

witness_preemption_handler = BiasedPreemptions(
    actions=witness_preemptions, bias=0.1, prefix="__witness_"
)


pinned_augmented = {
    "__evaluated_sally_throws": 0.0,  # consider sally not throwing
    "__antecedent_bill_hits": 1.0,  # not including bill hitting as an antecedent
    "__witness_bill_hits": 1.0,  # and witness preempting on bill hitting
}


# moved all interventions to the top, as suggested

with MultiWorldCounterfactual() as mwc:
    with (
        do(actions=antecedent_candidate_counterfactual),
        do(actions=evaluated_node_counterfactual),
        evaluated_node_preemption_handler,
        antecedent_node_preemption_handler,
        witness_preemption_handler,
        condition(
            data=pinned_augmented,
        ),
    ):
        # inverting the order of the context managers fails too
        # with (
        #     condition(
        #         data=pinned_augmented,
        #     ),
        #     witness_preemption_handler,
        #     antecedent_node_preemption_handler,
        #     evaluated_node_preemption_handler,
        #     do(actions=evaluated_node_counterfactual),
        #     do(actions=antecedent_candidate_counterfactual),
        # ):
        with pyro.poutine.trace() as trace:
            stones_bayesian_model()

tr = trace.trace.nodes


# Processing trace________________________


nodes_of_interest = [
    "sally_throws",
    "sally_hits",
    "bill_hits",
    "bottle_shatters",
]

values_table = {}
with mwc:
    for node in nodes_of_interest:
        value = tr[node]["value"]
        indices = indices_of(value)
        values_table[f"obs_{node}"] = (
            gather(
                value,
                IndexSet(**{intervened: {0} for intervened in indices.keys()}),
            )
            .squeeze()
            .tolist()
        )

        values_table[f"int_{node}"] = (
            gather(
                value,
                IndexSet(**{intervened: {1} for intervened in indices.keys()}),
            )
            .squeeze()
            .tolist()
        )

        if f"__evaluated_{node}" in tr.keys():
            values_table[f"__evaluated_{node}"] = (
                gather(
                    tr[f"__evaluated_{node}"]["value"],
                    IndexSet(**{intervened: {1} for intervened in indices.keys()}),
                )
                .squeeze()
                .tolist()
            )

        if f"__antecedent_{node}" in tr.keys():
            values_table[f"__antecedent_{node}"] = (
                gather(
                    tr[f"__antecedent_{node}"]["value"],
                    IndexSet(**{intervened: {1} for intervened in indices.keys()}),
                )
                .squeeze()
                .tolist()
            )

        if f"__witness_{node}" in tr.keys():
            values_table[f"__witness_{node}"] = (
                gather(
                    tr[f"__witness_{node}"]["value"],
                    IndexSet(**{intervened: {1} for intervened in indices.keys()}),
                )
                .squeeze()
                .tolist()
            )


print(values_table)

# Intended behavior________________________

assert (
    values_table["obs_sally_throws"] == 1.0
    and values_table["int_sally_throws"] == 0.0
    and values_table["obs_sally_hits"] == 1.0
    and values_table["int_sally_hits"] == 0.0
    and values_table["obs_bill_hits"] == 0.0
    and values_table["int_bill_hits"] == 0.0
    and values_table["obs_bottle_shatters"] == 1.0
    and values_table["int_bottle_shatters"] == 0.0
)
