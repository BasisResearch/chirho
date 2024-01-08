import collections.abc
import contextlib
from typing import Callable, Iterable, Mapping, TypeVar, Union

import pyro
import torch

from chirho.explainable.handlers.components import (
    consequent_differs,
    random_intervention,
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
    actions: Mapping[str, Intervention[T]],
    *,
    bias: float = 0.0,
    prefix: str = "__cause_split_",
):
    """
    A context manager used for a stochastic search of minimal but-for causes among potential interventions.
    On each run, nodes listed in `actions` are randomly selected and intervened on with probability `.5 + bias`
    (that is, preempted with probability `.5-bias`). The sampling is achieved by adding stochastic binary preemption
    nodes associated with intervention candidates. If a given preemption node has value `0`, the corresponding
    intervention is executed. See tests in `tests/explainable/test_split_subsets.py` for examples.

    :param actions: A mapping of sites to interventions.
    :param bias: The scalar bias towards not intervening. Must be between -0.5 and 0.5, defaults to 0.0.
    :param prefix: A prefix used for naming additional preemption nodes. Defaults to "__cause_split_".
    """
    # TODO support event_dim != 0 propagation in factual_preemption
    preemptions = {
        antecedent: undo_split(antecedents=[antecedent])
        for antecedent in actions.keys()
    }

    with do(actions=actions):
        with Preemptions(actions=preemptions, bias=bias, prefix=prefix):
            yield


@contextlib.contextmanager
def SearchForExplanation(
    antecedents: Union[
        Mapping[str, Intervention[T]],
        Mapping[str, pyro.distributions.constraints.Constraint],
    ],
    witnesses: Union[Mapping[str, Intervention[T]], Iterable[str]],
    consequents: Union[
        Mapping[str, Callable[[T], Union[float, torch.Tensor]]], Iterable[str]
    ],
    *,
    antecedent_bias: float = 0.0,
    witness_bias: float = 0.0,
    consequent_eps: float = -1e8,
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

    :param antecedents: A mapping from antecedent names to interventions.
    :param witnesses: A mapping from witness names to interventions.
    :param consequents: A mapping from consequent names to factor functions.
    """
    if isinstance(
        next(iter(antecedents.values())),
        pyro.distributions.constraints.Constraint,
    ):
        antecedents = {
            a: random_intervention(s, name=f"{antecedent_prefix}_proposal_{a}")
            for a, s in antecedents.items()
        }

    if not isinstance(witnesses, collections.abc.Mapping):
        witnesses = {
            w: undo_split(antecedents=list(antecedents.keys())) for w in witnesses
        }

    if not isinstance(consequents, collections.abc.Mapping):
        consequents = {
            c: consequent_differs(
                antecedents=list(antecedents.keys()), eps=consequent_eps
            )
            for c in consequents
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
        actions=antecedents, bias=antecedent_bias, prefix=antecedent_prefix
    )

    witness_handler = Preemptions(
        actions=witnesses, bias=witness_bias, prefix=witness_prefix
    )

    consequent_handler = Factors(factors=consequents, prefix=consequent_prefix)

    with antecedent_handler, witness_handler, consequent_handler:
        yield
