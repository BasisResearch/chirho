import functools
from typing import TypeVar

import pyro
import torch

S = TypeVar("S")
T = TypeVar("T")


@functools.singledispatch
def uniform_proposal(
    support: pyro.distributions.constraints.Constraint,
    **kwargs,
) -> pyro.distributions.Distribution:
    """
    This function heuristically constructs a probability distribution over a specified
    support. The choice of distribution depends on the type of support provided.

    - If the support is ``real``, it creates a wide Normal distribution
      and standard deviation, defaulting to ``(0,10)``.
    - If the support is ``boolean``, it creates a Bernoulli distribution with a fixed logit of ``0``,
      corresponding to success probability ``.5``.
    - If the support is an ``interval``, the transformed distribution is centered around the
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
