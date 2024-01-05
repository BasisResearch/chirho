import functools
from typing import Any, Callable, Iterable, Optional, Tuple, TypeVar

import pyro
import pyro.distributions as dist
import pyro.distributions.constraints as constraints
import torch
from torch.distributions import biject_to

from chirho.counterfactual.handlers.selection import get_factual_indices
from chirho.indexed.ops import IndexSet, cond, cond_n, gather
from chirho.interventional.ops import Intervention, intervene

S = TypeVar("S")
T = TypeVar("T", bound=Any)


@pyro.poutine.runtime.effectful(type="preempt")
@functools.partial(pyro.poutine.block, hide_types=["intervene"])
def preempt(
    obs: T, acts: Tuple[Intervention[T], ...], case: Optional[S] = None, **kwargs
) -> T:
    """
    Effectful primitive operation for "preempting" values in a probabilistic program.

    Unlike the counterfactual operation :func:`~chirho.counterfactual.ops.split`,
    which returns multiple values concatenated along a new axis
    via the operation :func:`~chirho.indexed.ops.scatter`,
    :func:`preempt` returns a single value determined by the argument ``case``
    via :func:`~chirho.indexed.ops.cond` .

    In a probabilistic program, a :func:`preempt` call induces a mixture distribution
    over downstream values, whereas :func:`split` would induce a joint distribution.

    :param obs: The observed value.
    :param acts: The interventions to apply.
    :param case: The case to select.
    """
    if case is None:
        return obs

    name = kwargs.get("name", None)
    act_values = {IndexSet(**{name: {0}}): obs}
    for i, act in enumerate(acts):
        act_values[IndexSet(**{name: {i + 1}})] = intervene(obs, act, **kwargs)

    return cond_n(act_values, case, event_dim=kwargs.get("event_dim", 0))


@functools.singledispatch
def soft_eq(support: constraints.Constraint, v1: T, v2: T, **kwargs) -> torch.Tensor:
    """
    Computes soft equality between two values `v1` and `v2` given a distribution constraint `support`.
    Returns a negative value if there is a difference (the larger the difference, the lower the value)
    and tends to `Norm(0,1).log_prob(0)` as `v1` and `v2` tend to each other,
    except for when the support is boolean, in which case it returns `0.0` if the values are equal
    and a large negative number (`eps`) otherwise.

    :param support: distribution constraint (`real`/`boolean`/`positive`/`interval`).
    :params v1, v2: the values to be compared.
    :param kwargs: Additional keywords arguments:
        - for boolean, the function expects `eps` to set a large negative value for.
        - For interval and real constraints, `scale` adjusts the softness of the inequality.
    :return: A tensor of log probabilities capturing the soft equality between `v1` and `v2`.
    :raises TypeError: If boolean tensors have different data types.
    """
    if support.is_discrete:
        raise NotImplementedError(
            "Soft equality is not implemented for arbitrary discrete distributions."
        )
    elif support is constraints.real:  # base case
        scale = kwargs.get("scale", 0.1)
        return dist.Normal(0.0, scale).log_prob(v1 - v2)
    else:
        # generative process:
        #   u1, u2 ~ base_dist
        #   v1 = tfm(u1), v2 = tfm(u2)
        #   ud = u1 - u2 ~ base_dist
        tfm = biject_to(support)
        v1_inv = tfm.inv(v1)
        ldj = tfm.log_abs_det_jacobian(v1_inv, v1)
        v2_inv = tfm.inv(v2)
        ldj = ldj + tfm.log_abs_det_jacobian(v2_inv, v2)
        for _ in range(tfm.codomain.event_dim - tfm.domain.event_dim):
            ldj = torch.sum(ldj, dim=-1)
        return soft_eq(tfm.domain, v1_inv, v2_inv, **kwargs) - ldj


@soft_eq.register
def _soft_eq_independent(support: constraints.independent, v1: T, v2: T, **kwargs):
    result = soft_eq(support.base_constraint, v1, v2, **kwargs)
    for _ in range(support.reinterpreted_batch_ndims):
        result = torch.sum(result, dim=-1)
    return result


@soft_eq.register(type(constraints.boolean))
def _soft_eq_boolean(support: constraints.Constraint, v1: T, v2: T, **kwargs):
    assert support is constraints.boolean
    scale = kwargs.get("scale", 0.1)
    return torch.log(cond(scale, 1 - scale, v1 == v2, event_dim=0))


@soft_eq.register
def _soft_eq_integer_interval(
    support: constraints.integer_interval, v1: T, v2: T, **kwargs
):
    scale = kwargs.get("scale", 0.1)
    width = support.upper_bound - support.lower_bound + 1
    return dist.Binomial(total_count=width, probs=scale).log_prob(torch.abs(v1 - v2))


@soft_eq.register(type(constraints.integer))
def _soft_eq_integer(support: constraints.Constraint, v1: T, v2: T, **kwargs):
    scale = kwargs.get("scale", 0.1)
    return dist.Poisson(rate=scale).log_prob(torch.abs(v1 - v2))


@soft_eq.register(type(constraints.positive_integer))
@soft_eq.register(type(constraints.nonnegative_integer))
def _soft_eq_positive_integer(support: constraints.Constraint, v1: T, v2: T, **kwargs):
    return soft_eq(constraints.integer, v1, v2, **kwargs)


@functools.singledispatch
def soft_neq(support: constraints.Constraint, v1: T, v2: T, **kwargs) -> torch.Tensor:
    """
    Computes soft inequality between two values `v1` and `v2` given a distribution constraint `support`.
    Tends to zero as the difference between the value increases, and tends to
    `-eps / (Norm(0,scale).log_prob(0) - 1e-10)` as `v1` and `v2` tend to each other.

    :param support: distribution constraint (`real`/`boolean`/`positive`/`interval`).
    :params v1, v2: the values to be compared.
    :param kwargs: Additional keywords arguments:
        - for boolean, the function expects `eps` to set a large negative value for.
        - For interval and real constraints, the function expects `eps` to fix the minimal value and
        `scale` to adjust the softness of the inequality.
    :return: A tensor of log probabilities capturing the soft inequality between `v1` and `v2`.
    :raises TypeError: If boolean tensors have different data types.
    :raises ValueError: If the specified scale is less than `1 / sqrt(2 * pi)`, to ensure that the log
                        probabilities used in calculations are are nonpositive.
    """
    if support.is_discrete:  # for discrete pmf, soft_neq = 1 - soft_eq (in log space)
        return torch.log(-torch.expm1(soft_eq(support, v1, v2, **kwargs)))
    elif support is constraints.real:  # base case
        scale = kwargs.get("scale", 0.1)
        return torch.log(2 * dist.Normal(0., scale).cdf(torch.abs(v1 - v2)) - 1)
    else:
        tfm = biject_to(support)
        return soft_neq(tfm.domain, tfm.inv(v1), tfm.inv(v2), **kwargs)


@soft_neq.register
def _soft_neq_independent(support: constraints.independent, v1: T, v2: T, **kwargs):
    result = soft_neq(support.base_constraint, v1, v2, **kwargs)
    for _ in range(support.reinterpreted_batch_ndims):
        result = torch.sum(result, dim=-1)
    return result


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
