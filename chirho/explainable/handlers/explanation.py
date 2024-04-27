import contextlib
from typing import Callable, Mapping, Tuple, TypeVar, Union

import pyro.distributions.constraints as constraints
import torch

from chirho.explainable.handlers.components import (  # sufficiency_intervention,
    consequent_eq,
    consequent_eq_neq,
    consequent_neq,
    random_intervention,
    sufficiency_intervention,
    undo_split,
)
from chirho.explainable.handlers.preemptions import Preemptions
from chirho.interventional.handlers import do
from chirho.interventional.ops import Intervention
from chirho.observational.handlers.condition import Factors

S = TypeVar("S")
T = TypeVar("T")


@contextlib.contextmanager
def SplitSubsets(
    supports: Mapping[str, constraints.Constraint],
    actions:  Mapping[ str, Union[Intervention[T], 
    Tuple[Intervention[T], ...]] ],
    #TODO deal with type-related linting errors
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

    with do(actions=actions):
        with Preemptions(actions=preemptions, bias=bias, prefix=prefix):
            yield


@contextlib.contextmanager
def SearchForExplanation(
    antecedents: Union[
        Mapping[str, Intervention[T]],
        Mapping[str, constraints.Constraint],
    ],
    witnesses: Union[
        Mapping[str, Intervention[T]], Mapping[str, constraints.Constraint]
    ],
    consequents: Union[
        Mapping[str, Callable[[T], Union[float, torch.Tensor]]],
        Mapping[str, constraints.Constraint],
    ],
    *,
    antecedent_bias: float = 0.0,
    witness_bias: float = 0.0,
    consequent_scale: float = 1e-2,
    antecedent_prefix: str = "__antecedent_",
    witness_prefix: str = "__witness_",
    consequent_prefix: str = "__consequent_",
):
    """
    Effect handler used for causal explanation search. On each run:

      1. The antecedent nodes are intervened on with the values in ``antecedents`` \
        using :func:`~chirho.counterfactual.ops.split` . \
        Unless alternative interventions are provided, \
        counterfactual values are uniformly sampled for each antecedent node \
        using :func:`~chirho.explainable.internals.uniform_proposal` \
        given its support as a :class:`~pyro.distributions.constraints.Constraint`.

      2. These interventions are randomly :func:`~chirho.explainable.ops.preempt`-ed \
        using :func:`~chirho.explainable.handlers.undo_split` \
        by a :func:`~chirho.explainable.handlers.SplitSubsets` handler.

      3. The witness nodes are randomly :func:`~chirho.explainable.ops.preempt`-ed \
        to be kept at the values given in ``witnesses``.

      4. A :func:`~pyro.factor` node is added tracking whether the consequent nodes differ \
        between the factual and counterfactual worlds.

    :param antecedents: A mapping from antecedent names to interventions or to constraints.
    :param witnesses: A mapping from witness names to interventions or to constraints.
    :param consequents: A mapping from consequent names to factor functions or to constraints.
    """
    if antecedents and isinstance(
        next(iter(antecedents.values())),
        constraints.Constraint,
    ):
        antecedents_supports = {a: s for a, s in antecedents.items()}
        antecedents = {
            a: random_intervention(s, name=f"{antecedent_prefix}_proposal_{a}")
            for a, s in antecedents.items()
        }
    else:
        antecedents_supports = {a: constraints.boolean for a in antecedents.keys()}
        # TODO generalize to non-scalar antecedents

    if witnesses and isinstance(
        next(iter(witnesses.values())),
        constraints.Constraint,
    ):
        witnesses = {
            w: undo_split(s, antecedents=list(antecedents.keys()))
            for w, s in witnesses.items()
        }

    if consequents and isinstance(
        next(iter(consequents.values())),
        constraints.Constraint,
    ):
        consequents = {
            c: consequent_neq(
                support=s,
                antecedents=list(antecedents.keys()),
                scale=consequent_scale,
            )
            for c, s in consequents.items()
        }

    if len(consequents) == 0:
        raise ValueError("must have at least one consequent")

    if len(antecedents) == 0:
        raise ValueError("must have at least one antecedent")

    if set(consequents.keys()) & set(antecedents.keys()):
        raise ValueError("consequents and possible antecedents must be disjoint")

    if set(consequents.keys()) & set(witnesses.keys()):
        raise ValueError("consequents and possible witnesses must be disjoint")

    antecedent_handler = SplitSubsets(
        supports=antecedents_supports,
        actions=antecedents,
        bias=antecedent_bias,
        prefix=antecedent_prefix,
    )

    witness_handler: Preemptions = Preemptions(
        actions=witnesses, bias=witness_bias, prefix=witness_prefix
    )

    consequent_handler = Factors(factors=consequents, prefix=consequent_prefix)

    with antecedent_handler, witness_handler, consequent_handler:
        yield


@contextlib.contextmanager
def SearchForNS(
    antecedents: Union[
        Mapping[str, Intervention[T]],
        Mapping[str, constraints.Constraint],
    ],
    witnesses: Union[
        Mapping[str, Intervention[T]], Mapping[str, constraints.Constraint]
    ],
    consequents: Union[
        Mapping[str, Callable[[T], Union[float, torch.Tensor]]],
        Mapping[str, constraints.Constraint],
    ],
    *,
    antecedent_bias: float = 0.0,
    witness_bias: float = 0.0,
    consequent_scale: float = 1e-2,
    antecedent_prefix: str = "__antecedent_",
    witness_prefix: str = "__witness_",
    consequent_prefix: str = "__consequent_",
):
    """
    Effect handler used for causal explanation search. On each run:

      1. The antecedent nodes are intervened on with the values in ``antecedents`` \
        using :func:`~chirho.counterfactual.ops.split` . \
        Unless alternative interventions are provided, \
        counterfactual values are uniformly sampled for each antecedent node \
        using :func:`~chirho.explainable.internals.uniform_proposal` \
        given its support as a :class:`~pyro.distributions.constraints.Constraint`.
        In another counterfactual world, the antecedent nodes are intervened to be
        at their factual values. The former will be used for probability-of-necessity-like
        calculations, while the latter will be used for the 
        probability-of-sufficiency-like ones.

      2. These interventions are randomly :func:`~chirho.explainable.ops.preempt`-ed \
        using :func:`~chirho.explainable.handlers.undo_split` \
        by a :func:`~chirho.explainable.handlers.SplitSubsets` handler.

      3. The witness nodes are randomly :func:`~chirho.explainable.ops.preempt`-ed \
        to be kept at the values given in ``witnesses``.

      4. A :func:`~pyro.factor` node is added tracking whether the consequent nodes differ \
        between the factual and counterfactual worlds.

    :param antecedents: A mapping from antecedent names to interventions or to constraints.
    :param witnesses: A mapping from witness names to interventions or to constraints.
    :param consequents: A mapping from consequent names to factor functions or to constraints.
    """
    antecedents_list = list(antecedents.keys())

    if antecedents and isinstance(
        next(iter(antecedents.values())),
        constraints.Constraint,
    ):

        antecedents_supports = {a: s for a, s in antecedents.items()}

        _antecedents = {
            a: (
                random_intervention(s, name=f"{antecedent_prefix}_proposal_{a}"),
                sufficiency_intervention(s, antecedents.keys()),
            )
            for a, s in antecedents_supports.items()
        }

    else:

        antecedents_supports = {a: constraints.boolean for a in antecedents.keys()}
        # TODO generalize to non-scalar antecedents
        # comment: how about extracting supports?

        _antecedents = {
            a: (antecedents[a], sufficiency_intervention(s, antecedents.keys()))
            for a, s in antecedents_supports.items()
        }

    if witnesses and isinstance(
        next(iter(witnesses.values())),
        constraints.Constraint,
    ):
        witnesses = {
            w: undo_split(s, antecedents=list(antecedents.keys()))
            for w, s in witnesses.items()
        }

    if consequents and isinstance(
        next(iter(consequents.values())),
        constraints.Constraint,
    ):

        consequents_eq = {
            c: consequent_eq(
                support=s,
                antecedents=antecedents_list,
                scale=consequent_scale,  # TODO allow for different scales for neq and eq
            )
            for c, s in consequents.items()
        }

        consequents_neq = {
            c: consequent_neq(
                support=s,
                antecedents=antecedents_list,
                scale=consequent_scale,  # TODO allow for different scales for neq and eq
            )
            for c, s in consequents.items()
        }

        consequents_eq_neq = {
            c: consequent_eq_neq(
                support=s,
                antecedents=antecedents_list,
                scale=consequent_scale,  # TODO allow for different scales for neq and eq
            )
            for c, s in consequents.items()
        }

    if len(consequents_neq) == 0:
        raise ValueError("must have at least one consequent")

    if len(antecedents) == 0:
        raise ValueError("must have at least one antecedent")

    if set(consequents_neq.keys()) & set(antecedents.keys()):
        raise ValueError("consequents and possible antecedents must be disjoint")

    if set(consequents_neq.keys()) & set(witnesses.keys()):
        raise ValueError("consequents and possible witnesses must be disjoint")

    antecedent_handler = SplitSubsets(
        supports=antecedents_supports,
        actions=_antecedents,
        bias=antecedent_bias,
        prefix=antecedent_prefix,
    )

    witness_handler: Preemptions = Preemptions(
        actions=witnesses, bias=witness_bias, prefix=witness_prefix
    )

    consequent_eq_handler: Factors = Factors(
        factors=consequents_eq, prefix=f"{consequent_prefix}_eq_"
    )

    consequent_neq_handler: Factors = Factors(
        factors=consequents_neq, prefix=f"{consequent_prefix}_neq_"
    )

    consequent_eq_neq_handler: Factors = Factors(
        factors=consequents_eq_neq, prefix=f"{consequent_prefix}_eq_neq_"
    )

    with (
        antecedent_handler
    ), (
        witness_handler
    ), consequent_neq_handler, consequent_eq_handler, consequent_eq_neq_handler:
        yield
