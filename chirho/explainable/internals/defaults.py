import functools
from typing import TypeVar

import pyro
import pyro.distributions as dist
import pyro.distributions.constraints as constraints
import torch
from torch.distributions import biject_to

from chirho.indexed.ops import cond

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


@functools.singledispatch
def soft_eq(support: constraints.Constraint, v1: T, v2: T, **kwargs) -> torch.Tensor:
    """
    Computes soft equality between two values ``v1`` and ``v2`` given a distribution constraint ``support``.
    Returns a negative value if there is a difference (the larger the difference, the lower the value)
    and tends to a low value as ``v1`` and ``v2`` tend to each other.

    :param support: A distribution constraint.
    :params v1, v2: the values to be compared.
    :param kwargs: Additional keywords arguments passed further; `scale` adjusts the softness of the inequality.
    :return: A tensor of log probabilities capturing the soft equality between ``v1`` and ``v2``,
            depends on the support and scale.
    :raises TypeError: If boolean tensors have different data types.

    Comment: if the support is boolean, setting ``scale = 1e-8`` results in a value close to ``0.0`` if the values
                are equal and a large negative number ``<=1e-8`` otherwise.
    """
    if not isinstance(v1, torch.Tensor) or not isinstance(v2, torch.Tensor):
        raise NotImplementedError("Soft equality is only implemented for tensors.")
    elif support.is_discrete:
        raise NotImplementedError(
            "Soft equality is not implemented for arbitrary discrete distributions."
        )
    elif support is constraints.real:  # base case
        scale = kwargs.get("scale", 0.1)
        return dist.Normal(0.0, scale).log_prob(v1 - v2)
    else:
        tfm = biject_to(support)
        v1_inv = tfm.inv(v1)
        ldj = tfm.log_abs_det_jacobian(v1_inv, v1)
        v2_inv = tfm.inv(v2)
        ldj = ldj + tfm.log_abs_det_jacobian(v2_inv, v2)
        for _ in range(tfm.codomain.event_dim - tfm.domain.event_dim):
            ldj = torch.sum(ldj, dim=-1)
        return soft_eq(tfm.domain, v1_inv, v2_inv, **kwargs) + ldj


@soft_eq.register
def _soft_eq_independent(support: constraints.independent, v1: T, v2: T, **kwargs):
    result = soft_eq(support.base_constraint, v1, v2, **kwargs)
    for _ in range(support.reinterpreted_batch_ndims):
        result = torch.sum(result, dim=-1)
    return result


@soft_eq.register(type(constraints.boolean))
def _soft_eq_boolean(support, v1: torch.Tensor, v2: torch.Tensor, **kwargs):
    assert support is constraints.boolean
    scale = kwargs.get("scale", 0.1)
    return torch.log(cond(scale, 1 - scale, v1 == v2, event_dim=0))


@soft_eq.register
def _soft_eq_integer_interval(
    support: constraints.integer_interval, v1: torch.Tensor, v2: torch.Tensor, **kwargs
):
    scale = kwargs.get("scale", 0.1)
    width = support.upper_bound - support.lower_bound + 1
    return dist.Binomial(total_count=width, probs=scale).log_prob(torch.abs(v1 - v2))


@soft_eq.register(type(constraints.integer))
def _soft_eq_integer(support, v1: torch.Tensor, v2: torch.Tensor, **kwargs):
    scale = kwargs.get("scale", 0.1)
    return dist.Poisson(rate=scale).log_prob(torch.abs(v1 - v2))


@soft_eq.register(type(constraints.positive_integer))
@soft_eq.register(type(constraints.nonnegative_integer))
def _soft_eq_positive_integer(support, v1: T, v2: T, **kwargs):
    return soft_eq(constraints.integer, v1, v2, **kwargs)


@functools.singledispatch
def soft_neq(support: constraints.Constraint, v1: T, v2: T, **kwargs) -> torch.Tensor:
    """
    Computes soft inequality between two values `v1` and `v2` given a distribution constraint `support`.
    Tends to `1-log(scale)` as the difference between the value increases, and tends to
    `log(scale)` as `v1` and `v2` tend to each other, summing elementwise over tensors.

    :param support: A distribution constraint.
    :params v1, v2: the values to be compared.
    :param kwargs: Additional keywords arguments:
        `scale` to adjust the softness of the inequality.
    :return: A tensor of log probabilities capturing the soft inequality between `v1` and `v2`.
    :raises TypeError: If boolean tensors have different data types.
    :raises NotImplementedError: If arguments are not tensors.
    """
    if not isinstance(v1, torch.Tensor) or not isinstance(v2, torch.Tensor):
        raise NotImplementedError("Soft equality is only implemented for tensors.")
    elif support.is_discrete:  # for discrete pmf, soft_neq = 1 - soft_eq (in log space)
        return torch.log(-torch.expm1(soft_eq(support, v1, v2, **kwargs)))
    elif support is constraints.real:  # base case
        scale = kwargs.get("scale", 0.1)
        return torch.log(2 * dist.Normal(0.0, scale).cdf(torch.abs(v1 - v2)) - 1)
    else:
        tfm = biject_to(support)
        return soft_neq(tfm.domain, tfm.inv(v1), tfm.inv(v2), **kwargs)


@soft_neq.register
def _soft_neq_independent(support: constraints.independent, v1: T, v2: T, **kwargs):
    result = soft_neq(support.base_constraint, v1, v2, **kwargs)
    for _ in range(support.reinterpreted_batch_ndims):
        result = torch.sum(result, dim=-1)
    return result
