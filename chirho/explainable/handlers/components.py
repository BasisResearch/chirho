import itertools
from typing import Callable, Iterable, MutableMapping, TypeVar

import pyro
import pyro.distributions.constraints as constraints
import torch

from chirho.counterfactual.handlers.selection import get_factual_indices
from chirho.explainable.internals import uniform_proposal
from chirho.indexed.ops import IndexSet, gather, indices_of, scatter_n

# from chirho.interventional.ops import intervene
from chirho.observational.handlers import soft_eq, soft_neq

S = TypeVar("S")
T = TypeVar("T")


def sufficiency_intervention(
    support: constraints.Constraint,
    antecedents: Iterable[str] = [],
) -> Callable[[T], T]:
    """
    Creates a sufficiency intervention for a single sample site, determined by
    the site name, intervening to keep the value as in the factual world with
    respect to the antecedents.

    :param support: The support constraint for the site.
    :param name: The sample site name.

    :return: A function that takes a `torch.Tensor` as input
        and returns the factual value at the named site as a tensor.

    Example::

        >>> with MultiWorldCounterfactual() as mwc:
        >>>     value = pyro.sample("value", proposal_dist)
        >>>     intervention = sufficiency_intervention(support)
        >>>     value = intervene(value, intervention)
    """

    def _sufficiency_intervention(value: T) -> T:

        indices = IndexSet(
            **{
                name: ind
                for name, ind in get_factual_indices().items()
                if name in antecedents
            }
        )

        factual_value = gather(
            value,
            indices,
            event_dim=support.event_dim,
        )
        return factual_value

    return _sufficiency_intervention


def random_intervention(
    support: constraints.Constraint,
    name: str,
) -> Callable[[T], T]:
    """
    Creates a random-valued intervention for a single sample site, determined by
    by the distribution support, and site name.

    :param support: The support constraint for the sample site.
    :param name: The name of the auxiliary sample site.

    :return: A function that takes a `torch.Tensor` as input
        and returns a random sample over the pre-specified support of the same
        event shape as the input tensor.

    Example::

        >>> support = pyro.distributions.constraints.real
        >>> intervention_fn = random_intervention(support, name="random_value")
        >>> with chirho.interventional.handlers.do(actions={"x": intervention_fn}):
        ...   x = pyro.deterministic("x", torch.tensor(2.))
        >>> assert x != 2
    """

    def _random_intervention(value: T) -> T:

        event_shape = value.shape[len(value.shape) - support.event_dim :]  # type: ignore

        proposal_dist = uniform_proposal(
            support,
            event_shape=event_shape,
        )
        return pyro.sample(name, proposal_dist)

    return _random_intervention


def undo_split(
    support: constraints.Constraint, antecedents: Iterable[str] = []
) -> Callable[[T], T]:
    """
    A helper function that undoes an upstream :func:`~chirho.counterfactual.ops.split` operation,
    meant to be used to create arguments to pass to :func:`~chirho.interventional.ops.intervene` ,
    :func:`~chirho.counterfactual.ops.split`  or :func:`~chirho.explainable.ops.preempt`.
    Works by gathering the factual value and scattering it back into two alternative cases.

    :param support: The support constraint for the site at which :func:`split` is being undone.
    :param antecedents: A list of upstream intervened sites which induced the :func:`split`
        to be reversed.
    :return: A callable that applied to a site value object returns a site value object in which
        the factual value has been scattered back into two alternative cases.
    """

    def _undo_split(value: T) -> T:
        antecedents_ = [
            a
            for a in antecedents
            if a in indices_of(value, event_dim=support.event_dim)
        ]

        factual_value = gather(
            value,
            IndexSet(**{antecedent: {0} for antecedent in antecedents_}),
            event_dim=support.event_dim,
        )

        # TODO exponential in len(antecedents) - add an indexed.ops.expand to do this cheaply
        return scatter_n(
            {
                IndexSet(
                    **{antecedent: {ind} for antecedent, ind in zip(antecedents_, inds)}
                ): factual_value
                for inds in itertools.product(*[[0, 1]] * len(antecedents_))
            },
            event_dim=support.event_dim,
        )

    return _undo_split


def consequent_eq(
    support: constraints.Constraint,
    antecedents: Iterable[str] = [],
    **kwargs,
) -> Callable[[T], torch.Tensor]:
    """
    A helper function for assessing whether values at a site are close to their observed values, assigning
    a small negative value close to zero if a value is close to its observed state and a large negative value otherwise.

    :param support: The support constraint for the consequent site.
    :param antecedents: A list of names of upstream intervened sites to consider when assessing similarity.

    :return: A callable which applied to a site value object (``consequent``), returns a tensor where each
             element indicates the extent to which the corresponding element of ``consequent``
             is close to its factual value.
    """

    def _consequent_eq(consequent: T) -> torch.Tensor:
        indices = IndexSet(
            **{
                name: ind
                for name, ind in get_factual_indices().items()
                if name in antecedents
            }
        )
        eq = soft_eq(
            support,
            consequent,
            gather(consequent, indices, event_dim=support.event_dim),
            **kwargs,
        )
        return eq

    return _consequent_eq


def consequent_neq(
    support: constraints.Constraint,
    antecedents: Iterable[str] = [],
    **kwargs,
) -> Callable[[T], torch.Tensor]:
    """
    A helper function for assessing whether values at a site differ from their observed values, assigning
    a small negative value close to zero if a value differs from its observed state
    and a large negative value otherwise.

    :param support: The support constraint for the consequent site.
    :param antecedents: A list of names of upstream intervened sites to consider when assessing differences.

    :return: A callable which applied to a site value object (``consequent``), returns a tensor where each
             element indicates whether the corresponding element of ``consequent`` differs from its factual value.
    """

    def _consequent_neq(consequent: T) -> torch.Tensor:
        indices = IndexSet(
            **{
                name: ind
                for name, ind in get_factual_indices().items()
                if name in antecedents
            }
        )
        diff = soft_neq(
            support,
            consequent,
            gather(consequent, indices, event_dim=support.event_dim),
            **kwargs,
        )
        return diff

    return _consequent_neq


def consequent_eq_neq(
    support: constraints.Constraint,
    antecedents: Iterable[str] = [],
    **kwargs,
) -> Callable[[T], torch.Tensor]:
    """
    A helper function for obtaining joint log prob of necessity and sufficiency. Assumes that
    the necessity intervention has been applied in counterfactual world 1 and sufficiency intervention in
    counterfactual world 2 (these can be passed as kwargs).

    :param support: The support constraint for the consequent site.
    :param antecedents: A list of names of upstream intervened sites to consider when composing the joint log prob.

    :return: A callable which applied to a site value object (``consequent``), returns a tensor with log prob sums
    of values resulting from necessity and sufficiency interventions, in appropriate counterfactual worlds.
    """

    def _consequent_eq_neq(consequent: T) -> torch.Tensor:

        factual_indices = IndexSet(
            **{
                name: ind
                for name, ind in get_factual_indices().items()
                if name in antecedents
            }
        )

        necessity_world = kwargs.get("necessity_world", 1)
        sufficiency_world = kwargs.get("sufficiency_world", 2)

        necessity_indices = IndexSet(
            **{
                name: {necessity_world}
                for name in indices_of(consequent, event_dim=support.event_dim).keys()
                if name in antecedents
            }
        )

        sufficiency_indices = IndexSet(
            **{
                name: {sufficiency_world}
                for name in indices_of(consequent, event_dim=support.event_dim).keys()
                if name in antecedents
            }
        )

        factual_value = gather(consequent, factual_indices, event_dim=support.event_dim)
        necessity_value = gather(
            consequent, necessity_indices, event_dim=support.event_dim
        )
        sufficiency_value = gather(
            consequent, sufficiency_indices, event_dim=support.event_dim
        )

        necessity_log_probs = soft_neq(
            support, necessity_value, factual_value, **kwargs
        )
        sufficiency_log_probs = soft_eq(
            support, sufficiency_value, factual_value, **kwargs
        )

        # nec_suff_log_probs = torch.add(necessity_log_probs, sufficiency_log_probs)

        FACTUAL_NEC_SUFF = torch.zeros_like(sufficiency_log_probs)
        # TODO reflect on this, do we want zeros?

        nec_suff_log_probs_partitioned = {
            **{
                factual_indices: FACTUAL_NEC_SUFF,
            },
            **{
                IndexSet(**{antecedent: {ind}}): log_prob
                for antecedent in (
                    set(antecedents)
                    & set(indices_of(consequent, event_dim=support.event_dim))
                )
                for ind, log_prob in zip(
                    [necessity_world, sufficiency_world],
                    [necessity_log_probs, sufficiency_log_probs],
                )
            },
        }

        new_value = scatter_n(
            nec_suff_log_probs_partitioned,
            event_dim=0,
        )

        return new_value

    return _consequent_eq_neq


class ExtractSupports(pyro.poutine.messenger.Messenger):
    """
    A Pyro Messenger for inferring distribution constraints.

    :return: An instance of ``ExtractSupports`` with a new attribute: ``supports``,
            a dictionary mapping variable names to constraints for all variables in the model.

    Example:

        >>> def mixed_supports_model():
        >>>     uniform_var = pyro.sample("uniform_var", dist.Uniform(1, 10))
        >>>     normal_var = pyro.sample("normal_var", dist.Normal(3, 15))
        >>> with ExtractSupports() as s:
        ...      mixed_supports_model()
        >>> print(s.supports)
    """

    supports: MutableMapping[str, pyro.distributions.constraints.Constraint]

    def __init__(self):
        super(ExtractSupports, self).__init__()

        self.supports = {}

    def _pyro_post_sample(self, msg: dict) -> None:
        if not pyro.poutine.util.site_is_subsample(msg):
            self.supports[msg["name"]] = msg["fn"].support
