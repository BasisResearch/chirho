from __future__ import annotations

import collections.abc
import contextlib
import functools
import itertools
from typing import Callable, Iterable, Mapping, TypeVar, Union

import pyro
import torch

from chirho.counterfactual.handlers.counterfactual import Preemptions
from chirho.counterfactual.handlers.selection import get_factual_indices
from chirho.indexed.ops import IndexSet, cond, gather, indices_of, scatter_n
from chirho.interventional.handlers import do
from chirho.interventional.ops import Intervention
from chirho.observational.handlers.condition import Factors

S = TypeVar("S")
T = TypeVar("T")


def undo_split(antecedents: Iterable[str] = [], event_dim: int = 0) -> Callable[[T], T]:
    """
    A helper function that undoes an upstream :func:`~chirho.counterfactual.ops.split` operation,
    meant to be used to create arguments to pass to :func:`~chirho.interventional.ops.intervene` ,
    :func:`~chirho.counterfactual.ops.split`  or :func:`~chirho.counterfactual.ops.preempt`.
    Works by gathering the factual value and scattering it back into two alternative cases.

    :param antecedents: A list of upstream intervened sites which induced the :func:`split` to be reversed.
    :param event_dim: The event dimension of the value to be preempted.
    :return: A callable that applied to a site value object returns a site value object in which
        the factual value has been scattered back into two alternative cases.
    """

    def _undo_split(value: T) -> T:
        antecedents_ = [
            a for a in antecedents if a in indices_of(value, event_dim=event_dim)
        ]

        factual_value = gather(
            value,
            IndexSet(**{antecedent: {0} for antecedent in antecedents_}),
            event_dim=event_dim,
        )

        # TODO exponential in len(antecedents) - add an indexed.ops.expand to do this cheaply
        return scatter_n(
            {
                IndexSet(
                    **{antecedent: {ind} for antecedent, ind in zip(antecedents_, inds)}
                ): factual_value
                for inds in itertools.product(*[[0, 1]] * len(antecedents_))
            },
            event_dim=event_dim,
        )

    return _undo_split


def consequent_differs(
    antecedents: Iterable[str] = [], eps: float = -1e8, event_dim: int = 0
) -> Callable[[T], torch.Tensor]:
    """
    A helper function for assessing whether values at a site differ from their observed values, assigning
    `eps` if a value differs from its observed state and `0.0` otherwise.

    :param antecedents: A list of names of upstream intervened sites to consider when assessing differences.
    :param eps: A numerical value assigned if the values differ, defaults to -1e8.
    :param event_dim: The event dimension of the value object.

    :return: A callable which applied to a site value object (`consequent`), returns a tensor where each
             element indicates whether the corresponding element of `consequent` differs from its factual value
             (`eps` if there is a difference, `0.0` otherwise).
    """

    def _consequent_differs(consequent: T) -> torch.Tensor:
        indices = IndexSet(
            **{
                name: ind
                for name, ind in get_factual_indices().items()
                if name in antecedents
            }
        )
        not_eq: torch.Tensor = consequent != gather(
            consequent, indices, event_dim=event_dim
        )
        for _ in range(event_dim):
            not_eq = torch.all(not_eq, dim=-1, keepdim=False)
        return cond(eps, 0.0, not_eq, event_dim=event_dim)

    return _consequent_differs


@functools.singledispatch
def uniform_proposal(
    support: pyro.distributions.constraints.Constraint,
    **kwargs,
) -> pyro.distributions.Distribution:
    """
    This function heuristically constructs a probability distribution over a specified
    support. The choice of distribution depends on the type of support provided.

    - If the support is `real`, it creates a wide Normal distribution
      and standard deviation, defaulting to (0,100).
    - If the support is `boolean`, it creates a Bernoulli distribution with a fixed logit of 0,
      corresponding to success probability .5.
    - If the support is an `interval`, the transformed distribution is centered around the
      midpoint of the interval.

    :param support: The support used to create the probability distribution.
    :param kwargs: Additional keyword arguments.
    :return: A uniform probability distribution over the specified support.
    """
    if support is pyro.distributions.constraints.real:
        return pyro.distributions.Normal(0, 10).mask(False)
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
    return d.expand(event_shape).to_event(support.reinterpreted_batch_ndims)


@uniform_proposal.register
def _uniform_proposal_integer(
    support: pyro.distributions.constraints.integer_interval,
    **kwargs,
) -> pyro.distributions.Distribution:
    if support.lower_bound != 0:
        raise NotImplementedError(
            "integer_interval with lower_bound > 0 not yet supported"
        )
    n = support.upper_bound - support.lower_bound + 1
    return pyro.distributions.Categorical(probs=torch.ones((n,)))


def random_intervention(
    support: pyro.distributions.constraints.Constraint,
    name: str,
) -> Callable[[torch.Tensor], torch.Tensor]:
    """
    Creates a random-valued intervention for a single sample site, determined by
    by the distribution support, and site name.

    :param support: The support constraint for the sample site.
    :param name: The name of the auxiliary sample site.

    :return: A function that takes a ``torch.Tensor`` as input
        and returns a random sample over the pre-specified support of the same
        event shape as the input tensor.

    Example::

        >>> support = pyro.distributions.constraints.real
        >>> intervention_fn = random_intervention(support, name="random_value")
        >>> with chirho.interventional.handlers.do(actions={"x": intervention_fn}):
        ...   x = pyro.deterministic("x", torch.tensor(2.))
        >>> assert x != 2
    """

    def _random_intervention(value: torch.Tensor) -> torch.Tensor:
        event_shape = value.shape[len(value.shape) - support.event_dim :]
        proposal_dist = uniform_proposal(
            support,
            event_shape=event_shape,
        )
        return pyro.sample(name, proposal_dist)

    return _random_intervention


@contextlib.contextmanager
def SearchForCause(
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
    intervention is executed. See tests in `tests/counterfactual/test_handlers_explanation.py` for examples.

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
def ExplainCauses(
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
        using :func:`~chirho.counterfactual.handlers.explanation.uniform_proposal` \
        given its support as a :class:`~pyro.distributions.constraints.Constraint` .

      2. These interventions are randomly :func:`~chirho.counterfactual.ops.preempt`-ed \
        using :func:`~chirho.counterfactual.handlers.explanation.undo_split` \
        by a :func:`~chirho.counterfactual.handlers.explanation.SearchForCause` handler.

      3. The witness nodes are randomly :func:`~chirho.counterfactual.ops.preempt`-ed \
        to be kept at the values given in ``witnesses`` .

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

    antecedent_handler = SearchForCause(
        actions=antecedents, bias=antecedent_bias, prefix=antecedent_prefix
    )

    witness_handler = Preemptions(
        actions=witnesses, bias=witness_bias, prefix=witness_prefix
    )

    consequent_handler = Factors(factors=consequents, prefix=consequent_prefix)

    with antecedent_handler, witness_handler, consequent_handler:
        yield
