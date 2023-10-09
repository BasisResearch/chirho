import functools
import itertools
from typing import Callable, Iterable, TypeVar

import pyro
import torch  # noqa: F401

from chirho.counterfactual.handlers.selection import get_factual_indices
from chirho.indexed.ops import IndexSet, cond, gather, indices_of, scatter

S = TypeVar("S")
T = TypeVar("T")


def undo_split(antecedents: Iterable[str] = [], event_dim: int = 0) -> Callable[[T], T]:
    """
    A helper function that undoes an upstream :func:`~chirho.counterfactual.ops.split` operation,
    meant to meant to be used to create arguments to pass to :func:`~chirho.interventional.ops.intervene` ,
    :func:`~chirho.counterfactual.ops.split`  or :func:`~chirho.counterfactual.ops.preempt` .
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
        return scatter(
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
        return pyro.distributions.Normal(0, 100).mask(False)
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
    """
    This constructs a probability distribution with independent dimensions
    over a specified support. The choice of distribution depends on the type of support provided
    (see the documentation for `uniform_proposal`).

    :param support: The support used to create the probability distribution.
    :param event_shape: The event shape specifying the dimensions of the distribution.
    :param kwargs: Additional keyword arguments.
    :return: A probability distribution with independent dimensions over the specified support.

    Example:
    ```
    indep_constraint = pyro.distributions.constraints.independent(
    pyro.distributions.constraints.real, reinterpreted_batch_ndims=2)
    dist = uniform_proposal(indep_constraint, event_shape=torch.Size([2, 3]))
    with pyro.plate("data", 3):
        samples_indep = pyro.sample("samples_indep", dist.expand([4, 2, 3]))
    ```
    """

    d = uniform_proposal(support.base_constraint, event_shape=event_shape, **kwargs)
    return d.expand(event_shape).to_event(support.reinterpreted_batch_ndims)


@uniform_proposal.register
def _uniform_proposal_integer(
    support: pyro.distributions.constraints.integer_interval,
    **kwargs,
) -> pyro.distributions.Distribution:
    """
    This constructs a uniform categorical distribution over an integer_interval support
    where the lower bound is 0 and the upper bound is specified by the support.

    :param support: The integer_interval support with a lower bound of 0 and a specified upper bound.
    :param kwargs: Additional keyword arguments.
    :return: A categorical probability distribution over the specified integer_interval support.

    Example:
    ```
    constraint = pyro.distributions.constraints.integer_interval(0, 2)
    dist = _uniform_proposal_integer(constraint)
    samples = dist.sample(torch.Size([100]))
    print(dist.probs.tolist())
    ```
    """
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
    Creates a random `pyro`sample` function for a single sample site, determined by
    by the distribution support, and site name.

    :param support: The support constraint for the sample site..can take.
    :param name: The name of the sample site.

    :return: A `pyro.sample` function that takes a torch.Tensor as input
        and returns a random sample over the pre-specified support of the same
        event shape as the input tensor.

    Example:
    ```
    support = pyro.distributions.constraints.real
    name = "real_sample"
    intervention_fn = random_intervention(support, name)
    random_sample = intervention_fn(torch.tensor(2.0))
    ```
    """

    def _random_intervention(value: torch.Tensor) -> torch.Tensor:
        event_shape = value.shape[len(value.shape) - support.event_dim :]
        proposal_dist = uniform_proposal(
            support,
            event_shape=event_shape,
        )
        return pyro.sample(name, proposal_dist)

    return _random_intervention
