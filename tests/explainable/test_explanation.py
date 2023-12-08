import pyro
import pyro.distributions as dist
import torch

from chirho.counterfactual.handlers.counterfactual import MultiWorldCounterfactual
from chirho.explainable.handlers import SearchForExplanation
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
    witnesses = ["bill_throws", "bill_hits"]
    consequents = ["bottle_shatters"]

    with MultiWorldCounterfactual() as mwc:
        with SearchForExplanation(
            antecedents=antecedents,
            witnesses=witnesses,
            consequents=consequents,
            antecedent_bias=0.1,
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

        assert all(lp == -1e8 or lp == 0.0 for lp in log_probs)

        for step in range(200):
            bottle_will_shatter = (
                st_obs[step] != st_int[step] and st_ant == 0.0
            ) or bh_int[step] == 1.0
            if bottle_will_shatter:
                assert log_probs[step] == -1e8
