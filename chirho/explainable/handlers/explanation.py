import contextlib
from typing import Callable, Mapping, Optional, TypeVar, Union

import pyro.distributions.constraints as constraints
import torch

from chirho.explainable.handlers.components import (
    consequent_eq_neq,
    random_intervention,
    sufficiency_intervention,
    undo_split,
)
from chirho.explainable.handlers.preemptions import Preemptions
from chirho.interventional.handlers import Interventions
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

    with Interventions(actions=actions):  # type: ignore
        with Preemptions(actions=preemptions, bias=bias, prefix=prefix):
            yield


def ExplanationTransform(
    supports: Mapping[str, constraints.Constraint],
    antecedent_alternatives: Optional[Mapping[str, Intervention[S]]] = None,
    consequent_factors: Optional[Mapping[str, Callable[[T], torch.Tensor]]] = None,
    witness_preemptions: Optional[
        Mapping[str, Union[Intervention[S], Intervention[T]]]
    ] = None,
    *,
    antecedent_bias: float = 0.0,
    consequent_scale: float = 1e-2,
    witness_bias: float = 0.0,
    antecedent_prefix: str = "__antecedent_",
    witness_prefix: str = "__witness_",
    consequent_prefix: str = "__consequent_",
):
    """
    A context manager used for a stochastic search of minimal but-for causes among potential interventions.
    On each run, nodes listed in ``actions`` are randomly selected and intervened on with probability ``.5 + bias``
    (that is, preempted with probability ``.5-bias``). The sampling is achieved by adding stochastic binary preemption
    nodes associated with intervention candidates. If a given preemption node has value ``0``, the corresponding
    intervention is executed. See tests in `tests/explainable/test_handlers_explanation.py` for examples.

    :param supports: A mapping of sites to their support constraints.
    :param antecedent_alternatives: A mapping of antecedent names to interventions.
    :param consequent_factors: A mapping of consequent names to factor functions.
    :param witness_preemptions: A mapping of witness names to interventions.
    :param antecedent_bias: The scalar bias towards not intervening. Must be between -0.5 and 0.5, defaults to 0.0.
    :param consequent_scale: The scale of the consequent factor functions, defaults to 1e-2.
    :param witness_bias: The scalar bias towards not intervening on witnesses. Must be between -0.5 and 0.5, defaults to 0.0.
    :param antecedent_prefix: A prefix used for naming additional antecedent nodes. Defaults to ``__antecedent_``.
    :param witness_prefix: A prefix used for naming additional witness nodes. Defaults to ``__witness_``.
    :param consequent_prefix: A prefix used for naming additional consequent nodes. Defaults to ``__consequent_``.

    :return: A context manager that can be used to query the evidence.
    """

    @contextlib.contextmanager
    def _query(
        antecedents: Mapping[str, Optional[Observation[S]]],
        consequents: Mapping[str, Optional[Observation[T]]],
    ):
        assert len(antecedents) > 0
        assert len(consequents) > 0
        assert not set(consequents.keys()) & set(antecedents.keys())
        assert set(antecedents.keys()) <= set(supports.keys())
        assert set(consequents.keys()) <= set(supports.keys())

        # default argument values
        if antecedent_alternatives is None:
            necessity_interventions = {
                a: random_intervention(supports[a], name=f"__antecedent_proposal_{a}")
                for a in antecedents.keys()
            }
        else:
            necessity_interventions = {
                a: antecedent_alternatives[a] for a in antecedents.keys()
            }

        sufficiency_interventions = {
            a: (
                aa
                if aa is not None
                else sufficiency_intervention(
                    supports[a], antecedents=antecedents.keys()
                )
            )
            for a, aa in antecedents.items()
        }

        interventions = {
            a: (necessity_interventions[a], sufficiency_interventions[a])
            for a in antecedents.keys()
        }

        if consequent_factors is None:
            factors = {
                c: consequent_eq_neq(
                    support=supports[c],
                    antecedents=antecedents.keys(),
                    scale=consequent_scale,
                )
                for c in consequents.keys()
            }
        else:
            factors = {c: consequent_factors[c] for c in consequents.keys()}

        if witness_preemptions is None:
            preemptions = {
                w: undo_split(supports[w], antecedents=antecedents.keys())
                for w in set(supports.keys()) - set(consequents.keys())
            }
        else:
            preemptions = {
                w: witness_preemptions[w]
                for w in set(supports.keys()) - set(consequents.keys())
            }

        antecedent_handler = SplitSubsets(
            {a: supports[a] for a in antecedents.keys()},
            interventions,
            bias=antecedent_bias,
            prefix=antecedent_prefix,
        )
        consequent_handler = Factors(factors, prefix=consequent_prefix)
        witness_handler = Preemptions(
            preemptions, bias=witness_bias, prefix=witness_prefix
        )
        with antecedent_handler, witness_handler, consequent_handler:
            evidence: Mapping[str, Union[Observation[S], Observation[T]]] = {
                **{a: aa for a, aa in antecedents.items() if aa is not None},
                **{c: cc for c, cc in consequents.items() if cc is not None},
            }
            yield evidence

    return _query
