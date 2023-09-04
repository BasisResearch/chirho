import collections.abc
import contextlib
import functools
from typing import Callable, Iterable, Mapping, ParamSpec, TypeVar

import pyro
import pyro.distributions
import torch

from chirho.counterfactual.handlers.counterfactual import BiasedPreemptions
from chirho.counterfactual.handlers.selection import get_factual_indices
from chirho.indexed.ops import IndexSet, cond, gather, indices_of, scatter
from chirho.interventional.handlers import do
from chirho.interventional.ops import Intervention
from chirho.observational.handlers.condition import Factors

P = ParamSpec("P")
T = TypeVar("T")


def undo_split(
    antecedents: Iterable[str] = [], event_dim: int = 0
) -> Callable[[T], T]:
    def _undo_split(value: T) -> T:
        antecedents_ = [
            a for a in antecedents if a in indices_of(value, event_dim=event_dim)
        ]

        factual_value = gather(
            value,
            IndexSet(**{antecedent: {0} for antecedent in antecedents_}),
            event_dim=event_dim,
        )

        return scatter(
            {
                IndexSet(
                    **{antecedent: {0} for antecedent in antecedents_}
                ): factual_value,
                IndexSet(
                    **{antecedent: {1} for antecedent in antecedents_}
                ): factual_value,
            },
            event_dim=event_dim,
        )

    return _undo_split


@contextlib.contextmanager
def PartOfCause(
    actions: Mapping[str, Intervention[T]],
    *,
    bias: float = 0.0,
    prefix: str = "__cause_split_",
):
    # TODO support event_dim != 0 propagation in factual_preemption
    preemptions = {
        antecedent: undo_split(antecedents=[antecedent])
        for antecedent in actions.keys()
    }

    with do(actions=actions):
        with BiasedPreemptions(actions=preemptions, bias=bias, prefix=prefix):
            with pyro.poutine.trace() as logging_tr:
                yield logging_tr.trace


def consequent_differs(
    antecedents: Iterable[str] = [], eps: float = -1e8, event_dim: int = 0
) -> Callable[[T], torch.Tensor]:

    def _consequent_differs(consequent: T) -> torch.Tensor:
        indices = IndexSet(**{
            name: ind for name, ind in get_factual_indices().items()
            if name in antecedents
        })
        not_eq = consequent != gather(consequent, indices, event_dim=event_dim)
        return cond(eps, 0.0, not_eq, event_dim=event_dim)

    return _consequent_differs


@functools.singledispatch
def uniform_proposal(
    support: pyro.distributions.constraints.Constraint, **kwargs,
) -> pyro.distributions.Distribution:
    """
    Heuristic for creating a uniform distribution on a given support.
    """
    if support is pyro.distributions.constraints.real:
        return pyro.distributions.Normal(0., 1.).mask(False)
    elif support is pyro.distributions.constraints.boolean:
        return pyro.distributions.Bernoulli(logits=torch.zeros(()))
    else:
        tfm = pyro.distributions.transforms.biject_to(support)
        base = uniform_proposal(pyro.distributions.constraints.real, **kwargs)
        return pyro.distributions.TransformedDistribution(base, tfm)


@uniform_proposal.register
def _uniform_proposal_indep(
    support: pyro.distributions.constraints.independent,
    *,
    event_shape: torch.Size = torch.Size([]),
    **kwargs,
) -> pyro.distributions.Distribution:
    d = uniform_proposal(support.base_constraint, event_shape=event_shape, **kwargs)
    return d.expand(kwargs["event_shape"]).to_event(support.reinterpreted_batch_ndims)


@uniform_proposal.register
def _uniform_proposal_integer(
    support: pyro.distributions.constraints.integer_interval, **kwargs,
) -> pyro.distributions.Distribution:
    if support.lower_bound != 0:
        raise NotImplementedError("non-zero lower bound not yet supported")
    n = support.upper_bound - support.lower_bound + 1
    return pyro.distributions.Categorical(probs=torch.ones((n,)))


def random_intervention(
    support: pyro.distributions.constraints.Constraint,
    name: str,
) -> Callable[[torch.Tensor], torch.Tensor]:
    """
    Creates a simple random intervention on a single site.

    :param support: The support of the sample site to be intervened on.
    :param name: The name of the sample site to be intervened on.
    """
    def _random_intervention(value: torch.Tensor) -> torch.Tensor:
        event_shape = value.shape[len(value.shape) - support.event_dim:]
        proposal_dist = uniform_proposal(support, event_shape=event_shape)
        return pyro.sample(name, proposal_dist)

    return _random_intervention


@contextlib.contextmanager
def ExplainCauses(
    antecedents: Mapping[str, Intervention[T]] | Mapping[str, pyro.distributions.constraints.Constraint],
    witnesses: Mapping[str, Intervention[T]] | Iterable[str],
    consequents: Mapping[str, Callable[[T], float | torch.Tensor]] | Iterable[str],
    *,
    antecedent_bias: float = 0.0,
    witness_bias: float = 0.0,
    consequent_eps: float = -1e8,
    antecedent_prefix: str = "__antecedent_",
    witness_prefix: str = "__witness_",
    consequent_prefix: str = "__consequent_",
):
    """
    Effect handler for causal explanation.

    :param antecedents: A mapping from antecedent names to interventions.
    :param witnesses: A mapping from witness names to interventions.
    :param consequents: A mapping from consequent names to factor functions.
    """
    if isinstance(next(iter(antecedents.values())), pyro.distributions.constraints.Constraint):
        antecedents = {
            a: random_intervention(s, name=f"{antecedent_prefix}_proposal_{a}")
            for a, s in antecedents.items()
        }

    if not isinstance(witnesses, collections.abc.Mapping):
        witnesses = {w: undo_split(antecedents=list(antecedents.keys())) for w in witnesses}

    if not isinstance(consequents, collections.abc.Mapping):
        consequents = {
            c: consequent_differs(antecedents=list(antecedents.keys()), eps=consequent_eps)
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

    antecedent_handler = PartOfCause(actions=antecedents, bias=antecedent_bias, prefix=antecedent_prefix)
    witness_handler = BiasedPreemptions(actions=witnesses, bias=witness_bias, prefix=witness_prefix)
    consequent_handler = Factors(factors=consequents, prefix=consequent_prefix)

    with antecedent_handler, witness_handler, consequent_handler:
        with pyro.poutine.trace() as logging_tr:
            yield logging_tr.trace
