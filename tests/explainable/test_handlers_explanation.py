import math

import pyro
import pyro.distributions as dist
import pyro.distributions.constraints as constraints
import torch

from chirho.counterfactual.handlers.counterfactual import MultiWorldCounterfactual
from chirho.explainable.handlers.components import undo_split
from chirho.explainable.handlers.explanation import SearchForExplanation, SplitSubsets
from chirho.explainable.handlers.preemptions import Preemptions
from chirho.indexed.ops import IndexSet, gather
from chirho.observational.handlers.condition import condition


def stones_bayesian_model():
    with pyro.poutine.mask(mask=False):
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


def test_SearchForExplanation():
    observations = {
        "prob_sally_throws": 1.0,
        "prob_bill_throws": 1.0,
        "prob_sally_hits": 1.0,
        "prob_bill_hits": 1.0,
        "prob_bottle_shatters_if_sally": 1.0,
        "prob_bottle_shatters_if_bill": 1.0,
    }

    observations_conditioning = condition(
        data={k: torch.as_tensor(v) for k, v in observations.items()}
    )

    antecedents = {"sally_throws": 0.0}
    witnesses = {"bill_throws": constraints.boolean, "bill_hits": constraints.boolean}
    consequents = {"bottle_shatters": constraints.boolean}

    with MultiWorldCounterfactual() as mwc:
        with SearchForExplanation(
            antecedents=antecedents,
            witnesses=witnesses,
            consequents=consequents,
            antecedent_bias=0.1,
            consequent_scale=1e-8,
        ):
            with observations_conditioning:
                with pyro.plate("sample", 200):
                    with pyro.poutine.trace() as tr:
                        stones_bayesian_model()

    tr.trace.compute_log_prob()
    tr = tr.trace.nodes

    with mwc:
        log_probs = (
            gather(
                tr["__consequent_bottle_shatters"]["log_prob"],
                IndexSet(**{i: {1} for i in antecedents.keys()}),
                event_dim=0,
            )
            .squeeze()
            .tolist()
        )

        st_obs = (
            gather(
                tr["sally_throws"]["value"],
                IndexSet(**{i: {0} for i in antecedents.keys()}),
                event_dim=0,
            )
            .squeeze()
            .tolist()
        )

        st_int = (
            gather(
                tr["sally_throws"]["value"],
                IndexSet(**{i: {1} for i in antecedents.keys()}),
                event_dim=0,
            )
            .squeeze()
            .tolist()
        )

        bh_int = (
            gather(
                tr["bill_hits"]["value"],
                IndexSet(**{i: {1} for i in antecedents.keys()}),
                event_dim=0,
            )
            .squeeze()
            .tolist()
        )

        st_ant = tr["__antecedent_sally_throws"]["value"].squeeze().tolist()

        assert all(lp <= -1e5 or lp > math.log(0.5) for lp in log_probs)

        for step in range(200):
            bottle_will_shatter = (
                st_obs[step] != st_int[step] and st_ant == 0.0
            ) or bh_int[step] == 1.0
            if bottle_will_shatter:
                assert log_probs[step] <= -1e5

    witnesses = {}
    with MultiWorldCounterfactual():
        with SearchForExplanation(
            antecedents=antecedents,
            witnesses=witnesses,
            consequents=consequents,
            antecedent_bias=0.1,
            consequent_scale=1e-8,
        ):
            with observations_conditioning:
                with pyro.plate("sample", 200):
                    with pyro.poutine.trace() as tr_empty:
                        stones_bayesian_model()

    assert tr_empty.trace.nodes


test_SearchForExplanation()


def test_SplitSubsets_single_layer():
    observations = {
        "prob_sally_throws": 1.0,
        "prob_bill_throws": 1.0,
        "prob_sally_hits": 1.0,
        "prob_bill_hits": 1.0,
        "prob_bottle_shatters_if_sally": 1.0,
        "prob_bottle_shatters_if_bill": 1.0,
    }

    observations_conditioning = condition(
        data={k: torch.as_tensor(v) for k, v in observations.items()}
    )

    with MultiWorldCounterfactual() as mwc:
        with SplitSubsets(
            supports={"sally_throws": constraints.boolean},
            actions={"sally_throws": 0.0},
            bias=0.0,
        ):
            with observations_conditioning:
                with pyro.poutine.trace() as tr:
                    stones_bayesian_model()

    tr = tr.trace.nodes

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


def test_SplitSubsets_two_layers():
    observations = {
        "prob_sally_throws": 1.0,
        "prob_bill_throws": 1.0,
        "prob_sally_hits": 1.0,
        "prob_bill_hits": 1.0,
        "prob_bottle_shatters_if_sally": 1.0,
        "prob_bottle_shatters_if_bill": 1.0,
    }

    observations_conditioning = condition(
        data={k: torch.as_tensor(v) for k, v in observations.items()}
    )

    actions = {"sally_throws": 0.0}

    pinned_preemption_variables = {
        "preempt_sally_throws": torch.tensor(0),
        "witness_preempt_bill_hits": torch.tensor(1),
    }
    preemption_conditioning = condition(data=pinned_preemption_variables)

    witness_preemptions = {
        "bill_hits": undo_split(constraints.boolean, antecedents=actions.keys())
    }
    witness_preemptions_handler: Preemptions = Preemptions(
        actions=witness_preemptions, prefix="witness_preempt_"
    )

    with MultiWorldCounterfactual() as mwc:
        with SplitSubsets(
            supports={"sally_throws": constraints.boolean},
            actions=actions,
            bias=0.1,
            prefix="preempt_",
        ):
            with preemption_conditioning, witness_preemptions_handler:
                with observations_conditioning:
                    with pyro.poutine.trace() as tr:
                        stones_bayesian_model()

    tr = tr.trace.nodes

    with mwc:
        obs_bill_hits = gather(
            tr["bill_hits"]["value"],
            IndexSet(**{"sally_throws": {0}}),
            event_dim=0,
        ).item()
        int_bill_hits = gather(
            tr["bill_hits"]["value"],
            IndexSet(**{"sally_throws": {1}}),
            event_dim=0,
        ).item()
        int_bottle_shatters = gather(
            tr["bottle_shatters"]["value"],
            IndexSet(**{"sally_throws": {1}}),
            event_dim=0,
        ).item()

    assert obs_bill_hits == 0.0 and int_bill_hits == 0.0 and int_bottle_shatters == 0.0
