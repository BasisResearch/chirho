import contextlib
from typing import (
    Callable,
    Iterable,
    Mapping,
    Optional,
    ParamSpec,
    TypeVar,
    List,
)

import pyro
import pyro.distributions as dist
import torch

from chirho.counterfactual.handlers.counterfactual import (
    BiasedPreemptions,
    MultiWorldCounterfactual,
)
from chirho.counterfactual.handlers.selection import get_factual_indices
from chirho.indexed.ops import IndexSet, cond, gather, indices_of, scatter
from chirho.interventional.handlers import do
from chirho.interventional.ops import Intervention
from chirho.observational.handlers import condition
from chirho.observational.handlers.condition import Factors
from chirho.observational.ops import Observation

P = ParamSpec("P")
T = TypeVar("T")

from chirho.counterfactual.handlers.explanation import (
    factual_preemption,
    consequent_differs_factor,
    PartOfCause,
    # Responsibility,
)


# ______________RESPONSIBILITY REVISED


@contextlib.contextmanager
def Responsibility(
    antecedent: Mapping[str, Intervention[torch.Tensor]],
    treatments: Mapping[str, Intervention[torch.Tensor]],
    witness_candidates: List[str],
    consequent: str,
    *,
    antecedent_bias: float = 0.0,
    treatment_bias: float = 0.0,
    witness_bias: float = 0.0,
):
    antecedent_handler = PartOfCause(
        antecedent, bias=antecedent_bias, prefix="__antecedent_"
    )

    treatment_handler = PartOfCause(
        treatments, bias=treatment_bias, prefix="__treatment_"
    )

    interventions = {**antecedent, **treatments}
    witness_preemptions = {
        candidate: factual_preemption(antecedents=interventions.keys())
        for candidate in witness_candidates
    }

    witness_handler = BiasedPreemptions(
        actions=witness_preemptions, bias=witness_bias, prefix="__witness_"
    )

    consequent_factor = {consequent: consequent_differs_factor()}
    consequent_handler = Factors(
        factors=consequent_factor, prefix="__consequent_"
    )

    with (
        antecedent_handler
    ), treatment_handler, witness_handler, consequent_handler:
        with pyro.poutine.trace() as tr:
            yield tr.trace.nodes


# _______________MODEL

# let's start with a minimal interesting example
# you are one of three voters


def voting_model():
    u_vote0 = pyro.sample("u_vote0", dist.Bernoulli(0.6))
    u_vote1 = pyro.sample("u_vote1", dist.Bernoulli(0.6))
    u_vote2 = pyro.sample("u_vote2", dist.Bernoulli(0.6))

    vote0 = pyro.deterministic("vote0", torch.tensor(u_vote0), event_dim=0)
    vote1 = pyro.deterministic("vote1", torch.tensor(u_vote1), event_dim=0)
    vote2 = pyro.deterministic("vote2", torch.tensor(u_vote2), event_dim=0)

    outcome = pyro.deterministic("outcome", vote0 + vote1 + vote2 > 1)

    return {"outcome": outcome.float()}


observations = dict(u_vote0=1.0, u_vote1=1.0, u_vote2=1.0)
voting_observed = condition(voting_model, data=observations)

# voting_observed()

# ____________PREP

antecedent = {"vote0": 0.0}
antecedent_bias = torch.tensor(0.0)

treatments = {"vote1": 0.0, "vote2": 0.0}
treatment_bias = torch.tensor(0.0)

witness_candidates = ["vote1", "vote2"]
# witness_preemptions = {
#     candidate: factual_preemption(antecedents=interventions.keys())
#     for candidate in witness_candidates
# }
witness_bias = torch.tensor(0.0)

consequent = "outcome"
# consequent_factor = {consequent: consequent_differs_factor()}

# antecedent_handler = PartOfCause(
#     antecedents, bias=antecedent_bias, prefix="__antecedent_"
# )

# treatment_handler = PartOfCause(
#     treatments, bias=treatment_bias, prefix="__treatment_"
# )

# witness_handler = BiasedPreemptions(
#     actions=witness_preemptions, bias=witness_bias, prefix="__witness_"
# )

# consequent_handler = Factors(factors=consequent_factor, prefix="__consequent_")


voting_tested = condition(
    voting_observed, data={"__treatment_vote1": torch.tensor(0.0)}
)
# ________________RUN
with MultiWorldCounterfactual() as mwc:
    # with (
    #    antecedent_handler
    # ), treatment_handler, witness_handler, consequent_handler:
    #    with pyro.poutine.trace() as tr:
    with Responsibility(
        antecedent,
        treatments,
        witness_candidates,
        consequent,
    ) as resp:
        voting_tested()
        # voting_observed()


# ________________ANALYSIS
print(resp.keys())

print("vote1:", resp["vote1"]["value"])

antecedent = "vote0"

# print(resp[f"__antecedent_{antecedent}"]["value"])


def get_table(mwc, nodes, antecedent, treatments, witness_candidates, outcome):
    interventions = [antecedent] + treatments
    values_table = {}

    with mwc:
        values_table[f"obs_{antecedent}"] = (
            gather(
                nodes[antecedent]["value"],
                IndexSet(**{_int: {0} for _int in interventions}),
                event_dim=0,
            )
            .squeeze()
            .tolist()
        )

        values_table[f"int_{antecedent}"] = (
            gather(
                nodes[antecedent]["value"],
                IndexSet(**{_int: {1} for _int in interventions}),
                event_dim=0,
            )
            .squeeze()
            .tolist()
        )

        values_table[f"apr_{antecedent}"] = (
            nodes[f"__antecedent_{antecedent}"]["value"].squeeze().tolist()
        )

        values_table[f"alp_{antecedent}"] = (
            nodes[f"__antecedent_{antecedent}"]["fn"]
            .log_prob(nodes[f"__antecedent_{antecedent}"]["value"])
            .squeeze()
            .tolist()
        )

        for treatment in treatments:
            values_table[f"obs_{treatment}"] = (
                gather(
                    nodes[treatment]["value"],
                    IndexSet(**{_int: {0} for _int in interventions}),
                    event_dim=0,
                )
                .squeeze()
                .tolist()
            )

            values_table[f"int_{treatment}"] = (
                gather(
                    nodes[treatment]["value"],
                    IndexSet(**{_int: {1} for _int in interventions}),
                    event_dim=0,
                )
                .squeeze()
                .tolist()
            )

            values_table[f"apr_{treatment}"] = (
                nodes[f"__treatment_{treatment}"]["value"].squeeze().tolist()
            )

            values_table[f"alp_{treatment}"] = (
                nodes[f"__treatment_{treatment}"]["fn"]
                .log_prob(nodes[f"__treatment_{treatment}"]["value"])
                .squeeze()
                .tolist()
            )

    # for antecedent in antecedents:
    #     values_table[f"obs_{antecedent}"] = nodes[antecedent]["value"][0].squeeze().tolist()
    #     values_table[f"int_{antecedent}"] = nodes[antecedent]["value"][1].squeeze().tolist()
    #     values_table['apr_' + antecedent] = nodes['__treatment_split_' + antecedent]["value"].squeeze().tolist()
    #     values_table['alp_' + antecedent] = nodes['__treatment_split_' + antecedent]["fn"].log_prob(nodes['__treatment_split_' + antecedent]["value"]).squeeze().tolist()

    #     if f"__witness_split_{antecedent}" in nodes.keys():
    #         values_table['wpr_' + antecedent] = nodes['__witness_split_' + antecedent]["value"].squeeze().tolist()
    #         values_table['wlp_' + antecedent] = nodes['__witness_split_' + antecedent]["fn"].log_prob(nodes['__witness_split_' + antecedent]["value"]).squeeze().tolist()

    # values_table['cdif'] = nodes['consequent_differs_binary']["value"].squeeze().tolist()
    # values_table['clp'] = nodes['consequent_differs']["fn"].log_prob(nodes['consequent_differs']["value"]).squeeze().tolist()

    # if isinstance(values_table['clp'], float):
    #     values_df = pd.DataFrame([values_table])
    # else:
    #     values_df = pd.DataFrame(values_table)

    # values_df = pd.DataFrame(values_table)

    # summands_ant = ['alp_' + antecedent for antecedent in antecedents]
    # summands_wit = ['wlp_' + witness for witness in witness_candidates]
    # summands = [f"elp_{evaluated_node}"] +  summands_ant + summands_wit + ['clp']

    # values_df["int"] =  values_df.apply(lambda row: sum(row[row.index.str.startswith("apr_")] == 0), axis=1)
    # values_df['int'] = 1 - values_df[f"epr_{evaluated_node}"] + values_df["int"]
    # values_df["wpr"] = values_df.apply(lambda row: sum(row[row.index.str.startswith("wpr_")] == 1), axis=1)
    # values_df["changes"] =   values_df["int"] + values_df["wpr"]

    # values_df["sum_lp"] =  values_df[summands].sum(axis = 1)
    # values_df.drop_duplicates(inplace = True)
    # values_df.sort_values(by = "sum_lp", inplace = True, ascending = False)

    # tab =  values_df.reset_index(drop = True)

    # tab = remove_redundant_rows(tab)

    tab = values_table

    return tab


print(
    get_table(
        mwc,
        resp,
        "vote0",
        ["vote1", "vote2"],
        witness_candidates,
        consequent,
    )
)
