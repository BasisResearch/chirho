import math

import pyro
import pyro.distributions as dist
import pyro.distributions.constraints as constraints
import torch
import pytest

from chirho.counterfactual.handlers.counterfactual import MultiWorldCounterfactual
from chirho.explainable.handlers.components import undo_split
from chirho.explainable.handlers.explanation import SearchForExplanation, SplitSubsets
from chirho.explainable.handlers import ExtractSupports
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
    supports = {
        "sally_throws": constraints.boolean,
        "bill_throws": constraints.boolean,
        "sally_hits": constraints.boolean,
        "bill_hits": constraints.boolean,
        "bottle_shatters": constraints.boolean,
    }

    antecedents = {"sally_throws": torch.tensor(1.0)}

    consequents = {"bottle_shatters": torch.tensor(1.0)}

    witnesses = {
        "bill_throws": None,
    }

    observation_keys = [
        "prob_sally_throws",
        "prob_bill_throws",
        "prob_sally_hits",
        "prob_bill_hits",
        "prob_bottle_shatters_if_sally",
        "prob_bottle_shatters_if_bill",
    ]
    observations = {k: torch.tensor(1.0) for k in observation_keys}

    observations_conditioning = condition(
        data={k: torch.as_tensor(v) for k, v in observations.items()}
    )

    alternatives = {"sally_throws": 0.0}

    with MultiWorldCounterfactual() as mwc:
        with SearchForExplanation(
            supports=supports,
            antecedents=antecedents,
            consequents=consequents,
            witnesses=witnesses,
            alternatives=alternatives,
            antecedent_bias=0.1,
            consequent_scale=1e-8,
        ):
            with observations_conditioning:
                with pyro.plate("sample", 20):
                    with pyro.poutine.trace() as tr:
                        stones_bayesian_model()

    tr.trace.compute_log_prob()
    tr = tr.trace.nodes

    with mwc:
        nec_log_probs = (
            gather(
                tr["__cause____consequent_bottle_shatters"]["log_prob"],
                IndexSet(**{i: {1} for i in antecedents.keys()}),
                event_dim=0,
            )
            .squeeze()
            .tolist()
        )

        suff_log_probs = (
            gather(
                tr["__cause____consequent_bottle_shatters"]["log_prob"],
                IndexSet(**{i: {2} for i in antecedents.keys()}),
                event_dim=0,
            )
            .squeeze()
            .tolist()
        )

        st_nec = (
            gather(
                tr["sally_throws"]["value"],
                IndexSet(**{i: {1} for i in antecedents.keys()}),
                event_dim=0,
            )
            .squeeze()
            .tolist()
        )

        bh_nec = (
            gather(
                tr["bill_hits"]["value"],
                IndexSet(**{i: {1} for i in antecedents.keys()}),
                event_dim=0,
            )
            .squeeze()
            .tolist()
        )

        st_suff = (
            gather(
                tr["sally_throws"]["value"],
                IndexSet(**{i: {2} for i in antecedents.keys()}),
                event_dim=0,
            )
            .squeeze()
            .tolist()
        )

        bh_suff = (
            gather(
                tr["bill_hits"]["value"],
                IndexSet(**{i: {2} for i in antecedents.keys()}),
                event_dim=0,
            )
            .squeeze()
            .tolist()
        )

        assert all(lp <= -1e5 or lp > math.log(0.5) for lp in nec_log_probs)
        assert all(lp <= -10 or lp == 0.0 for lp in suff_log_probs)

        for step in range(20):
            if st_nec[step] == 0 and bh_nec[step] == 0:
                assert nec_log_probs[step] == 0.0
            else:
                assert nec_log_probs[step] <= -1e10

            if st_suff[step] == 1 or bh_suff[step] == 1:
                assert suff_log_probs[step] == 0.0
            else:
                assert suff_log_probs[step] <= -10


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

def test_edge_eq_neq():
    def model_independent():
        X = pyro.sample("X", dist.Bernoulli(0.5))
        Y = pyro.sample("Y", dist.Bernoulli(0.5))

    def model_connected():
        X = pyro.sample("X", dist.Bernoulli(0.5))
        Y = pyro.sample("Y", dist.Bernoulli(X))

    with ExtractSupports() as supports_independent:
        model_independent()

    with ExtractSupports() as supports_connected:
        model_connected()

    with MultiWorldCounterfactual() as mwc_independent:  
            with SearchForExplanation(
                supports=supports_independent.supports,
                antecedents={"X": torch.tensor(1.0)},
                consequents={"Y": torch.tensor(1.0)},
                witnesses={},
                alternatives={"X": torch.tensor(0.0)},
                antecedent_bias=-0.5,
                consequent_scale=0,
            ):
                with pyro.plate("sample", size=3):
                    with pyro.poutine.trace() as trace_independent:
                        model_independent()

    with MultiWorldCounterfactual() as mwc_connected:  
            with SearchForExplanation(
                supports=supports_connected.supports,
                antecedents={"X": torch.tensor(1.0)},
                consequents={"Y": torch.tensor(1.0)},
                witnesses={},
                alternatives={"X": torch.tensor(0.0)},
                antecedent_bias=-0.5,
                consequent_scale=0,
            ):
                with pyro.plate("sample", size=3):
                    with pyro.poutine.trace() as trace_connected:
                        model_connected()

    with MultiWorldCounterfactual() as mwc_reverse:  
            with SearchForExplanation(
                supports=supports_connected.supports,
                antecedents={"Y": torch.tensor(1.0)},
                consequents={"X": torch.tensor(1.0)},
                witnesses={},
                alternatives={"Y": torch.tensor(0.0)},
                antecedent_bias=-0.5,
                consequent_scale=0,
            ):
                with pyro.plate("sample", size=3):
                    with pyro.poutine.trace() as trace_reverse:
                        model_connected()


    trace_connected.trace.compute_log_prob
    trace_independent.trace.compute_log_prob
    trace_reverse.trace.compute_log_prob

    Y_values_ind = trace_independent.trace.nodes["Y"]["value"]

    if torch.any(Y_values_ind == 1.):
        print("testing with ", Y_values_ind)
        assert trace_independent.trace.nodes["__cause____consequent_Y"]["fn"].log_factor[1,0,0,0,:].sum().exp() == 0.
    else:
        assert trace_independent.trace.nodes["__cause____consequent_Y"]["fn"].log_factor[1,0,0,0,:].sum().exp() == 1.

    assert torch.all(trace_independent.trace.nodes["__cause____consequent_Y"]["fn"].log_factor.sum().exp() == 0)

    if torch.any(Y_values_ind == 0.):
        assert trace_independent.trace.nodes["__cause____consequent_Y"]["fn"].log_factor[2,0,0,0,:].sum().exp() == 0.
    else:
        assert trace_independent.trace.nodes["__cause____consequent_Y"]["fn"].log_factor[2,0,0,0,:].sum().exp() == 1.

    Y_values_con = trace_connected.trace.nodes["Y"]["value"]
    assert torch.all(trace_connected.trace.nodes["__cause____consequent_Y"]["fn"].log_factor.sum() == 0)
        
    X_values_rev = trace_reverse.trace.nodes["X"]["value"]
    if torch.any(X_values_rev == 1.):
        print("testing with ", Y_values_ind)
        assert trace_reverse.trace.nodes["__cause____consequent_X"]["fn"].log_factor[1,0,0,0,:].sum().exp() == 0.
    else:
        assert trace_reverse.trace.nodes["__cause____consequent_X"]["fn"].log_factor[1,0,0,0,:].sum().exp() == 1.

    if torch.any(X_values_rev == 0.):
        assert trace_reverse.trace.nodes["__cause____consequent_X"]["fn"].log_factor[2,0,0,0,:].sum().exp() == 0.
    else:
        assert trace_reverse.trace.nodes["__cause____consequent_X"]["fn"].log_factor[2,0,0,0,:].sum().exp() == 1.

    assert torch.all(trace_reverse.trace.nodes["__cause____consequent_X"]["fn"].log_factor.sum().exp() == 0)

# X -> Z, Y -> Z
def model_three_converge():
    X = pyro.sample("X", dist.Bernoulli(0.5))
    Y = pyro.sample("Y", dist.Bernoulli(0.5))
    Z = pyro.sample("Z", dist.Bernoulli(torch.min(X, Y)))
    return {"X": X, "Y": Y, "Z": Z}

# X -> Y, X -> Z
def model_three_diverge():
    X = pyro.sample("X", dist.Bernoulli(0.5))
    Y = pyro.sample("Y", dist.Bernoulli(X))
    Z = pyro.sample("Z", dist.Bernoulli(X))
    return {"X": X, "Y": Y, "Z": Z}

# X -> Y -> Z
def model_three_chain():
    X = pyro.sample("X", dist.Bernoulli(0.5))
    Y = pyro.sample("Y", dist.Bernoulli(X))
    Z = pyro.sample("Z", dist.Bernoulli(Y))
    return {"X": X, "Y": Y, "Z": Z}

# X -> Y, X -> Z, Y -> Z
def model_three_complete():
    X = pyro.sample("X", dist.Bernoulli(0.5))
    Y = pyro.sample("Y", dist.Bernoulli(X))
    Z = pyro.sample("Z", dist.Bernoulli(torch.max(X, Y)))
    return {"X": X, "Y": Y, "Z": Z}

# X -> Y    Z
def model_three_isolate():
    X = pyro.sample("X", dist.Bernoulli(0.5))
    Y = pyro.sample("Y", dist.Bernoulli(X))
    Z = pyro.sample("Z", dist.Bernoulli(0.5))
    return {"X": X, "Y": Y, "Z": Z}

# X     Y    Z
def model_three_independent():
    X = pyro.sample("X", dist.Bernoulli(0.5))
    Y = pyro.sample("Y", dist.Bernoulli(0.5))
    Z = pyro.sample("Z", dist.Bernoulli(0.5))
    return {"X": X, "Y": Y, "Z": Z}

@pytest.mark.parametrize("ante_cons", [("X", "Y", "Z"), ("X", "Z", "Y")])
@pytest.mark.parametrize("model", [model_three_converge, model_three_diverge, model_three_chain, model_three_complete, model_three_isolate, model_three_independent])
def test_eq_neq_three_variables(model, ante_cons):
    ante1, ante2, cons = ante_cons    
    with ExtractSupports() as supports:
        model()

    with MultiWorldCounterfactual() as mwc: 
        with SearchForExplanation(
            supports=supports.supports,
            antecedents={ante1: torch.tensor(1.0), ante2: torch.tensor(1.0)},
            consequents={cons: torch.tensor(1.0)},
            witnesses={},
            alternatives={ante1: torch.tensor(0.0), ante2: torch.tensor(0.0)},
            antecedent_bias=-0.5,
            consequent_scale=0,
        ):
            with pyro.plate("sample", size=1):
                with pyro.poutine.trace() as trace:
                    model()

    trace.trace.compute_log_prob
    nodes = trace.trace.nodes

    values = nodes[cons]["value"]
    log_probs = nodes[f"__cause____consequent_{cons}"]["fn"].log_factor

    assert log_probs.shape == torch.Size([3, 3, 1, 1, 1, 1])

    fact_worlds = IndexSet(**{name : {0} for name in [ante1, ante2]})
    nec_worlds = IndexSet(**{name : {1} for name in [ante1, ante2]})
    suff_worlds = IndexSet(**{name : {2} for name in [ante1, ante2]})
    with mwc:
        fact_lp = gather(log_probs, fact_worlds)
        assert fact_lp.exp().item() == 1

        nec_value = gather(values, nec_worlds)
        nec_lp = gather(log_probs, nec_worlds)
        assert nec_lp.exp().item() == 1 - nec_value.item()

        suff_value = gather(values, suff_worlds)
        suff_lp = gather(log_probs, suff_worlds)
        assert suff_lp.exp().item() == suff_value.item()

    assert torch.allclose(log_probs.squeeze().fill_diagonal_(0.0), torch.tensor(0.0))
