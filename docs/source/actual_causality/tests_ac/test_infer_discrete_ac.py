# TEST Ideally, for models with categorical variables we'd like to be able
# to use infer_discrete instead of sampling to answer actual causality queries
# right now, it seems, its use does not return as possible combinations
# that should be the right answers (even if it is made to return them,
# we also need to make sure they are MAP estimates).

import torch

from torch.optim import Adam
from typing import Dict, List, Optional, Union, Callable, Any

import pandas as pd
import functools
import pytest

import pyro
from pyro.distributions import Bernoulli
from pyro.infer import infer_discrete

from chirho.indexed.ops import IndexSet, gather, indices_of, scatter
from chirho.observational.handlers import condition
from chirho.interventional.handlers import do
from chirho.counterfactual.handlers.counterfactual import (
    MultiWorldCounterfactual,
    Preemptions,
    BiasedPreemptions,
)


# ____________________________________________________________
# Effect handler for AC, same as the one in
# `pre_release_versions` folder
# on the `causality-staging` branch.


class HalpernPearlModifiedApproximate:
    def __init__(
        self,
        model: Callable,
        counterfactual_antecedents: Dict[str, torch.Tensor],
        outcome: str,
        witness_candidates: List[str],
        observations: Optional[Dict[str, torch.Tensor]] = None,
    ):
        if observations is None:
            observations = {}

        self.model = model
        self.counterfactual_antecedents = counterfactual_antecedents
        self.outcome = outcome
        self.witness_candidates = witness_candidates
        self.observations = observations

        self.antecedent_preemptions = {
            antecedent: functools.partial(
                self.preempt_with_factual, antecedents=[antecedent]
            )
            for antecedent in self.counterfactual_antecedents.keys()
        }

        self.witness_preemptions = {
            candidate: functools.partial(
                self.preempt_with_factual,
                antecedents=self.counterfactual_antecedents,
            )
            for candidate in self.witness_candidates
        }

    @staticmethod
    def preempt_with_factual(
        value: torch.Tensor,
        *,
        antecedents: List[str] = None,
        event_dim: int = 0
    ):
        if antecedents is None:
            antecedents = []

        antecedents = [
            a
            for a in antecedents
            if a in indices_of(value, event_dim=event_dim)
        ]

        factual_value = gather(
            value,
            IndexSet(**{antecedent: {0} for antecedent in antecedents}),
            event_dim=event_dim,
        )

        return scatter(
            {
                IndexSet(
                    **{antecedent: {0} for antecedent in antecedents}
                ): factual_value,
                IndexSet(
                    **{antecedent: {1} for antecedent in antecedents}
                ): factual_value,
            },
            event_dim=event_dim,
        )

    def __call__(self, *args, **kwargs):
        with MultiWorldCounterfactual():
            with do(actions=self.counterfactual_antecedents):
                # the last element of the tensor is the factual case (preempted)
                with BiasedPreemptions(
                    actions=self.antecedent_preemptions,
                    bias=0.1,
                    prefix="__split_",
                ):
                    with Preemptions(actions=self.witness_preemptions):
                        with condition(
                            data={
                                k: torch.as_tensor(v)
                                for k, v in self.observations.items()
                            }
                        ):
                            with pyro.poutine.trace() as self.trace:
                                self.consequent = self.model(*args, **kwargs)[
                                    self.outcome
                                ]
                                self.intervened_consequent = gather(
                                    self.consequent,
                                    IndexSet(
                                        **{
                                            ant: {1}
                                            for ant in self.counterfactual_antecedents
                                        }
                                    ),
                                )
                                self.observed_consequent = gather(
                                    self.consequent,
                                    IndexSet(
                                        **{
                                            ant: {0}
                                            for ant in self.counterfactual_antecedents
                                        }
                                    ),
                                )
                                self.consequent_differs = (
                                    self.intervened_consequent
                                    != self.observed_consequent
                                )
                                pyro.deterministic(
                                    "consequent_differs_binary",
                                    self.consequent_differs,
                                    event_dim=0,
                                )  # feels inelegant
                                pyro.factor(
                                    "consequent_differs",
                                    torch.where(
                                        self.consequent_differs,
                                        torch.tensor(0.0),
                                        torch.tensor(-1e8),
                                    ),
                                )


# _______________________________________________________________
# trace processing

# this will explore the trace once we run inference on the model


def get_table(nodes, antecedents, witness_candidates):
    values_table = {}

    for antecedent in antecedents:
        values_table[antecedent] = (
            nodes[antecedent]["value"].squeeze().tolist()
        )
        values_table["preempted_" + antecedent] = (
            nodes["__split_" + antecedent]["value"].squeeze().tolist()
        )
        values_table["preempted_" + antecedent + "_log_prob"] = (
            nodes["__split_" + antecedent]["fn"]
            .log_prob(nodes["__split_" + antecedent]["value"])
            .squeeze()
            .tolist()
        )

    for candidate in witness_candidates:
        _values = nodes[candidate]["value"].squeeze().tolist()
        values_table["fixed_factual_" + candidate] = (
            nodes["__split_" + candidate]["value"].squeeze().tolist()
        )

    values_table["consequent_differs_binary"] = (
        nodes["consequent_differs_binary"]["value"].squeeze().tolist()
    )
    values_table["consequent_log_prob"] = (
        nodes["consequent_differs"]["fn"]
        .log_prob(nodes["consequent_differs"]["value"])
        .squeeze()
        .tolist()
    )

    if isinstance(values_table["consequent_log_prob"], float):
        values_df = pd.DataFrame([values_table])
    else:
        values_df = pd.DataFrame(values_table)

    summands = [
        "preempted_" + antecedent + "_log_prob" for antecedent in antecedents
    ]
    summands.append("consequent_log_prob")
    values_df["sum_log_prob"] = values_df[summands].sum(axis=1)
    values_df.drop_duplicates(inplace=True)
    values_df.sort_values(by="sum_log_prob", inplace=True, ascending=False)

    return values_df.reset_index(drop=True)


# This uses the trace to answer the query


def ac_check(
    hpm,
    nodes,
    counterfactual_antecedents=None,
    witness_candidates=None,
    outcome=None,
):
    if counterfactual_antecedents is None:
        antecedents = list(hpm.counterfactual_antecedents.keys())
    else:
        antecedents = counterfactual_antecedents

    if witness_candidates is None:
        witness_candidates = hpm.witness_candidates

    if outcome is None:
        consequent = hpm.outcome

    table = get_table(nodes, antecedents, witness_candidates)

    if table["sum_log_prob"][0] <= -1e8:
        print("No resulting difference to the consequent in the sample.")
        return

    winner = table.iloc[0]
    active_antecedents = []
    for antecedent in antecedents:
        if winner["preempted_" + antecedent] == 0:
            active_antecedents.append(antecedent)

    ac_flag = set(active_antecedents) == set(antecedents)

    if not ac_flag:
        print("The antecedent set is not minimal.")
    else:
        print("The antecedent set is an actual cause.")

    return ac_flag


# ------------------------------------


# Simple model based on the bottle shattering example______
sally_hits_pt = torch.tensor([1.0])
bill_hits_cpt = torch.tensor([1.0, 0.0])
bottle_shatters_cpt = torch.tensor([[0.0, 1.0], [1.0, 1.0]])

probs = (sally_hits_pt, bill_hits_cpt, bottle_shatters_cpt)


@pyro.infer.config_enumerate
def bottle_bn(sally_hits_pt, bill_hits_cpt, bottle_shatters_cpt):
    sh = pyro.sample("sh", Bernoulli(sally_hits_pt)).long()
    bh = pyro.sample("bh", Bernoulli(bill_hits_cpt[sh])).long()
    bs = pyro.sample("bs", Bernoulli(bottle_shatters_cpt[sh, bh])).long()

    # bh = bh.float()
    # bh = bh.float()
    # bs = bs.float()

    return {"sh": sh, "bh": bh, "bs": bs}


def bottle_bn_complete():
    return bottle_bn(*probs)


bottle_bn_HPM = HalpernPearlModifiedApproximate(
    model=bottle_bn_complete,
    counterfactual_antecedents={"sh": 0.0},
    outcome="bs",
    witness_candidates=["bh"],
)

# making sure everything else works
# and sampling gives us the right answer___________
with pyro.poutine.trace() as basic_trace:
    with pyro.plate("runs", 100):
        bottle_bn_HPM()

btr = basic_trace.trace.nodes

# we're interested in the top row(s)
sampled_table = get_table(btr, antecedents=["sh"], witness_candidates=["bh"])
print(sampled_table)

print(ac_check(bottle_bn_HPM, btr) is True)


def test_sampled_answer():
    assert ac_check(bottle_bn_HPM, btr)


# __________________________________________________________
# expecting finding the same setting using enumeration but failing
# #

serving_model_HPM = infer_discrete(
    pyro.plate("runs", 50, dim=-1)(bottle_bn_HPM),
    first_available_dim=-9,
    temperature=1,  # for now, we want full enumeration
)

with pyro.poutine.trace() as serving_trace:
    serving_model_HPM()

str = serving_trace.trace.nodes


# seems like bh doesn't get preempted
print(get_table(str, antecedents=["sh"], witness_candidates=["bh"]))


print(
    ac_check(
        serving_model_HPM,
        str,
        counterfactual_antecedents={"sh": 0.0},
        witness_candidates=["bh"],
        outcome="bs",
    )
)


def test_enumerated_answer():
    assert (
        ac_check(
            serving_model_HPM,
            str,
            counterfactual_antecedents={"sh": 0.0},
            witness_candidates=["bh"],
            outcome="bs",
        )
        is True
    )


# testing overkill: setting tempertaure to 0
# for a MAP estimate doesn't help either
# ___________________________________________________
map_model_HPM = infer_discrete(
    pyro.plate("runs", 2)(bottle_bn_HPM), first_available_dim=-9, temperature=0
)

with pyro.poutine.trace() as map_trace:
    map_model_HPM()

mtr = map_trace.trace.nodes


# this for some reason does not seem to give the right answer:
# is it  perhaps because __split_bh 1 MAP has
# the same prob logs and only one get listed?
# or: witnesses don't get preempted for some reason?
print(get_table(mtr, antecedents=["sh"], witness_candidates=["bh"]))


def test_map_answer():
    assert (
        ac_check(
            map_model_HPM,
            mtr,
            counterfactual_antecedents={"sh": 0.0},
            witness_candidates=["bh"],
            outcome="bs",
        )
        is True
    )


# A somewhat unrelated problem:
# object attributes are not preserved under infer_discrete,
# which is a bit of an inconvenience
