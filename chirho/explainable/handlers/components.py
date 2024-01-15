import itertools
from typing import Callable, Iterable, MutableMapping, TypeVar

import pyro
import pyro.distributions.constraints as constraints
import torch

from chirho.counterfactual.handlers.selection import get_factual_indices
from chirho.explainable.internals import uniform_proposal
from chirho.indexed.ops import IndexSet, gather, indices_of, scatter_n
from chirho.observational.handlers import soft_neq

S = TypeVar("S")
T = TypeVar("T")


def random_intervention(
    support: constraints.Constraint,
    name: str,
) -> Callable[[torch.Tensor], torch.Tensor]:
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

    def _random_intervention(value: torch.Tensor) -> torch.Tensor:
        event_shape = value.shape[len(value.shape) - support.event_dim :]
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


def consequent_differs(
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

    def _consequent_differs(consequent: T) -> torch.Tensor:
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

    return _consequent_differs


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
