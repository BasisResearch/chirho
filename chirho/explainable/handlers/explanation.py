import contextlib
import warnings
from typing import Callable, Mapping, Optional, TypeVar, Union, cast

import pyro.distributions as dist
import pyro.distributions.constraints as constraints
import torch

from chirho.explainable.handlers.components import (
    consequent_eq_neq,
    proposal_intervention,
    random_intervention,
    sufficiency_intervention,
    undo_split,
)
from chirho.explainable.handlers.preemptions import Preemptions
from chirho.interventional.handlers import do
from chirho.interventional.ops import Intervention
from chirho.observational.handlers.condition import Factors
from chirho.observational.ops import Observation

S = TypeVar("S")
T = TypeVar("T")


@contextlib.contextmanager
def SplitSubsets(
    supports: Mapping[str, constraints.Constraint],
    actions: Mapping[str, Intervention[T]],
    *,
    bias: float = 0.0,
    prefix: str = "__cause_split_",
    cases: Optional[Mapping[str, torch.Tensor]] = None,
):
    """
    A context manager used for a stochastic search of minimal but-for causes among potential interventions.
    On each run, nodes listed in ``actions`` are randomly selected and intervened on with probability ``.5 + bias``
    (that is, preempted with probability ``.5-bias``). The sampling is achieved by adding stochastic binary preemption
    nodes associated with intervention candidates. If a given preemption node has value ``0``, the corresponding
    intervention is executed. See tests in `tests/explainable/test_handlers_explanation.py` for examples.

    :param supports: A mapping of sites to their support constraints.
    :param actions: A mapping of sites to interventions.
    :param bias: The scalar bias towards not intervening. Must be between -0.5 and 0.5, defaults to 0.0.
    :param prefix: A prefix used for naming additional preemption nodes. Defaults to ``__cause_split_``.
    """
    preemptions = {
        antecedent: undo_split(supports[antecedent], antecedents=[antecedent])
        for antecedent in actions.keys()
    }

    with do(actions=actions):
        with Preemptions(actions=preemptions, bias=bias, prefix=prefix, cases=cases):
            yield


@contextlib.contextmanager
def SearchForExplanation(
    supports: Mapping[str, constraints.Constraint],
    antecedents: Mapping[str, Optional[Observation[S]]],
    consequents: Mapping[str, Optional[Observation[T]]],
    witnesses: Optional[
        Mapping[str, Optional[Union[Observation[S], Observation[T]]]]
    ] = None,
    *,
    alternatives: Optional[Mapping[str, Intervention[S]]] = None,
    factors: Optional[Mapping[str, Callable[[T], torch.Tensor]]] = None,
    preemptions: Optional[Mapping[str, Union[Intervention[S], Intervention[T]]]] = None,
    proposal_distributions: Optional[
        Mapping[str, torch.distributions.Distribution]
    ] = None,
    consequent_scale: float = 1e-2,
    antecedent_bias: float = 0.0,
    witness_bias: float = 0.0,
    prefix: str = "__cause__",
    num_samples: Optional[int] = None,
    sampling_dim: Optional[int] = None,
):
    """
    A handler for transforming causal explanation queries into probabilistic inferences.

    When used as a context manager, ``SearchForExplanation`` yields a dictionary of observations
    that can be used with ``condition`` to simultaneously impose an additional factivity constraint
    alongside the necessity and sufficiency constraints implemented by ``SearchForExplanation`` ::

        with SearchForExplanation(supports, antecedents, consequents, ...) as evidence:
            with condition(data=evidence):
                model()

    :param supports: A mapping of sites to their support constraints.
    :param antecedents: A mapping of antecedent names to optional observations.
    :param consequents: A mapping of consequent names to optional observations.
    :param witnesses: A mapping of witness names to optional observations.
    :param alternatives: An optional mapping of names to alternative antecedent interventions.
    :param factors: An optional mapping of names to consequent constraint factors.
    :param preemptions: An optional mapping of names to witness preemption values.
    :param antecedent_bias: The scalar bias towards not intervening. Must be between -0.5 and 0.5, defaults to 0.0.
    :param consequent_scale: The scale of the consequent factor functions, defaults to 1e-2.
    :param witness_bias: The scalar bias towards not preempting. Must be between -0.5 and 0.5, defaults to 0.0.
    :param prefix: A prefix used for naming additional consequent nodes. Defaults to ``__consequent_``.

    :return: A context manager that can be used to query the evidence.
    """
    ########################################
    # Validate input arguments
    ########################################
    assert len(antecedents) > 0
    assert len(consequents) > 0
    assert not set(consequents.keys()) & set(antecedents.keys())
    assert set(antecedents.keys()) <= set(supports.keys())
    assert set(consequents.keys()) <= set(supports.keys())
    if witnesses is not None:
        assert set(witnesses.keys()) <= set(supports.keys())
        assert not set(witnesses.keys()) & set(consequents.keys())
    else:
        # if witness candidates are not provided, use all non-consequent nodes
        warnings.warn(
            "Witness candidates were not provided. Using all non-consequent nodes.",
            UserWarning,
        )

        witnesses = {w: None for w in set(supports.keys()) - set(consequents.keys())}

    ##################################################################
    # Fill in default argument values and create constituent handlers
    ##################################################################

    # defaults for necessity interventions
    # alternatives = (
    #     {a: alternatives[a] for a in antecedents.keys()}
    #     if alternatives is not None
    #     else {
    #         a: random_intervention(supports[a], name=f"{prefix}_alternative_{a}")
    #         for a in antecedents.keys()
    #     }
    # )

    # TODO: simplify?
    _alternatives = {}
    for a in antecedents.keys():
        if alternatives is not None and a in alternatives.keys():
            _alternatives[a] = alternatives[a]
        else:
            if (
                proposal_distributions is not None
                and a in proposal_distributions.keys()
            ):
                _alternatives[a] = cast(
                    Intervention[S],
                    proposal_intervention(
                        proposal_distributions[a], name=f"{prefix}_alternative_{a}"
                    ),
                )
            else:
                _alternatives[a] = random_intervention(
                    supports[a], name=f"{prefix}_alternative_{a}"
                )

    alternatives = _alternatives

    # alternatives = (
    #     {a: alternatives[a] for a in antecedents.keys()}
    #     if alternatives is not None and alternatives[a] is not None
    #     else {
    #         a: (
    #             cast(
    #                 Intervention[S],
    #                 proposal_intervention(
    #                     proposal_distributions[a], name=f"{prefix}_alternative_{a}"
    #                 ),
    #             )
    #             if proposal_distributions and a in proposal_distributions
    #             else random_intervention(supports[a], name=f"{prefix}_alternative_{a}")
    #         )
    #         for a in antecedents.keys()
    #     }
    # )

    print("alternatives", alternatives)

    # defaults for sufficiency interventions
    sufficiency_actions = {
        a: (
            antecedents[a]
            if antecedents[a] is not None
            else sufficiency_intervention(supports[a], antecedents=antecedents.keys())
        )
        for a in antecedents.keys()
    }

    antecedent_actions = {a: (alternatives[a], sufficiency_actions[a]) for a in antecedents.keys()}  # type: ignore

    if num_samples is not None:
        if sampling_dim is None:
            raise ValueError("sampling_dim must be provided if num_samples is provided")

        case_shape = [1] * torch.abs(torch.tensor(sampling_dim))
        case_shape[sampling_dim] = num_samples

        antecedent_probs = torch.tensor(
            [0.5 - antecedent_bias] + ([(0.5 + antecedent_bias)])
        )

        antecedent_case_dist = dist.Categorical(probs=antecedent_probs)

        antecedent_cases = {
            key: antecedent_case_dist.sample(case_shape) for key in antecedents.keys()
        }

        # for key in antecedents.keys():
        #     antecedent_cases[key] = antecedent_case_dist.sample(case_shape)

        witness_probs = torch.tensor([0.5 - witness_bias] + ([(0.5 + witness_bias)]))

        witness_case_dist = dist.Categorical(probs=witness_probs)

        witness_cases = {
            key: witness_case_dist.sample(case_shape) for key in witnesses.keys()
        }

        witness_cases = {
            key: value * antecedent_cases[key] if key in antecedent_cases else value
            for key, value in witness_cases.items()
        }

    # for key, value in antecedents.items():

    # interventions on subsets of antecedents
    antecedent_handler = SplitSubsets(
        supports={a: supports[a] for a in antecedents.keys()},
        actions=antecedent_actions,
        bias=antecedent_bias,
        prefix=f"{prefix}__antecedent_",
        cases=antecedent_cases if num_samples is not None else None,
    )

    # defaults for witness_preemptions

    _preemptions = {}
    for w in witnesses.keys():
        if witnesses[w] is not None:
            _preemptions[w] = witnesses[w]
        else:
            _preemptions[w] = undo_split(supports[w], antecedents=antecedents.keys())

    print("preemptions", _preemptions)

    witness_handler = Preemptions(
        _preemptions,
        bias=witness_bias,
        prefix=f"{prefix}__witness_",
        cases=witness_cases if num_samples is not None else None,
    )
    # witness_handler = Preemptions(
    #     (
    #         {w: preemptions[w] for w in witnesses}
    #         if preemptions is not None
    #         else {
    #             w: undo_split(supports[w], antecedents=antecedents.keys())
    #             for w in witnesses
    #         }
    #     ),
    #     bias=witness_bias,
    #     prefix=f"{prefix}__witness_",
    # )

    #
    consequent_handler: Factors[T] = Factors(
        (
            {c: factors[c] for c in consequents.keys()}
            if factors is not None
            else {
                c: consequent_eq_neq(
                    support=supports[c],
                    proposed_consequent=consequents[c],  # added this
                    antecedents=antecedents.keys(),
                    scale=consequent_scale,
                )
                for c in consequents.keys()
            }
        ),
        prefix=f"{prefix}__consequent_",
    )

    ######################################################################
    # Apply handlers and yield evidence for optional factual conditioning
    ######################################################################
    evidence: Mapping[str, Union[Observation[S], Observation[T]]] = {
        **{a: aa for a, aa in antecedents.items() if aa is not None},
        **{c: cc for c, cc in consequents.items() if cc is not None},
        **{w: ww for w, ww in (witnesses or {}).items() if ww is not None},
    }
    with antecedent_handler, witness_handler, consequent_handler:
        yield evidence
