from typing import TypeVar

import pyro
import pyro.distributions as dist
import pytest
import torch

from chirho.counterfactual.handlers.counterfactual import MultiWorldCounterfactual
from chirho.counterfactual.handlers.explanation import PartOfCause
from chirho.indexed.ops import IndexSet, cond, gather, indices_of, scatter
from chirho.observational.handlers import condition


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
        with PartOfCause({"sally_throws": 0.0}, prefix="preempt_"):
            with condition(
                data={k: torch.as_tensor(v) for k, v in observations.items()}
            ):
                with pyro.poutine.trace() as trace:
                    stones_bayesian_model()

    tr = trace.trace.nodes

    with mwc:
        preempt_sally_throws = gather(
            tr["preempt_sally_throws"]["value"],
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


def test_two_layer_stones():
    observations = {
        "prob_sally_throws": 1.0,
        "prob_bill_throws": 1.0,
        "prob_sally_hits": 1.0,
        "prob_bill_hits": 1.0,
        "prob_bottle_shatters_if_sally": 1.0,
        "prob_bottle_shatters_if_bill": 1.0,
    }

    evaluated_node_counterfactual = {"sally_throws": 0.0}
    witness_node_counterfactual = {"bill_hits": 0.0}

    with MultiWorldCounterfactual() as mwc:
        with PartOfCause(evaluated_node_counterfactual, prefix="preempt_"), PartOfCause(
            witness_node_counterfactual, prefix="witness_"
        ), condition(
            data={k: torch.as_tensor(v) for k, v in observations.items()}
        ), pyro.poutine.trace() as trace:
            stones_bayesian_model()

    tr = trace.trace.nodes

    print("preempt_sally_throws:", tr["preempt_sally_throws"]["value"])
    print("sally_throws:", tr["sally_throws"]["value"])

    print("witness_bill_hits:", tr["witness_bill_hits"]["value"])
    print("bill_hits:", tr["bill_hits"]["value"])

    print("bottle_shatters:", tr["bottle_shatters"]["value"])

    with mwc:
        preempt_sally_throws = tr["preempt_sally_throws"]["value"]
        int_sally_hits = gather(
            tr["sally_hits"]["value"],
            IndexSet(**{"sally_throws": {1}}),
            event_dim=0,
        )

        preempt_bill_hits = tr["witness_bill_hits"]["value"]
        obs_bill_hits = gather(
            tr["bill_hits"]["value"],
            IndexSet(**{"sally_throws": {0}}),
            event_dim=0,
        )
        int_bill_hits = gather(
            tr["bill_hits"]["value"],
            IndexSet(**{"sally_throws": {1}}),
            event_dim=0,
        )
        int_bottle_shatters = gather(
            tr["bottle_shatters"]["value"],
            IndexSet(**{"sally_throws": {1}}),
            event_dim=0,
        )

    outcome = {
        "preempt_sally_throws": preempt_sally_throws.item(),
        "int_sally_hits": int_sally_hits.item(),
        "preempt_bill_hits": preempt_bill_hits.item(),
        "obs_bill_hits": obs_bill_hits.item(),
        "int_bill_hits": int_bill_hits.item(),
        "intervened_bottle_shatters": int_bottle_shatters.item(),
    }

    if outcome["preempt_sally_throws"] == 0 and outcome["preempt_bill_hits"] == 1:
        raise Exception(
            "Intervened bill_hits is ",
            outcome["int_bill_hits"],
            " and it should be 0!",
        )
