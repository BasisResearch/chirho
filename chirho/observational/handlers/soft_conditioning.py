from __future__ import annotations

import functools
import operator
from typing import Callable, Literal, Optional, Protocol, TypedDict, TypeVar, Union

import pyro
import pyro.distributions as dist
import pyro.distributions.constraints as constraints
import torch
from torch.distributions import biject_to

from chirho.indexed.ops import cond

T = TypeVar("T")


Kernel = Callable[[T, T], torch.Tensor]


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
    Computes soft inequality between two values ``v1`` and ``v2`` given a distribution constraint ``support``.
    Tends to a small value near zero as the difference between the value increases, and tends to
    a large negative value as ``v1`` and ``v2`` tend to each other, summing elementwise over tensors.

    :param support: A distribution constraint.
    :params v1, v2: the values to be compared.
    :param kwargs: Additional keywords arguments:
        `scale` to adjust the softness of the inequality.
    :return: A tensor of log probabilities capturing the soft inequality between ``v1`` and ``v2``.
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


class TorchKernel(torch.nn.Module):
    support: constraints.Constraint

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class SoftEqKernel(TorchKernel):
    """
    Kernel that returns a Bernoulli log-probability of equality.
    """

    support: constraints.Constraint = constraints.boolean
    alpha: torch.Tensor

    def __init__(self, alpha: Union[float, torch.Tensor], *, event_dim: int = 0):
        super().__init__()
        self.register_buffer("alpha", torch.as_tensor(alpha))
        if event_dim > 0:
            self.support = constraints.independent(constraints.boolean, event_dim)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return soft_eq(self.support, x, y, scale=self.alpha)


class RBFKernel(TorchKernel):
    """
    Kernel that returns a Normal log-probability of distance.
    """

    support: constraints.Constraint = constraints.real
    scale: torch.Tensor

    def __init__(self, scale: Union[float, torch.Tensor], *, event_dim: int = 0):
        super().__init__()
        self.register_buffer("scale", torch.as_tensor(scale))
        if event_dim > 0:
            self.support = constraints.independent(constraints.real, event_dim)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return (
            pyro.distributions.Normal(loc=0.0, scale=self.scale)
            .expand([1] * self.support.event_dim)
            .to_event(self.support.event_dim)
            .log_prob(x - y)
        )


class _MaskedDelta(Protocol):
    base_dist: pyro.distributions.Delta
    event_dim: int
    _mask: Union[bool, torch.Tensor]


class _DeterministicReparamMessage(TypedDict):
    name: str
    fn: _MaskedDelta
    value: torch.Tensor
    is_observed: Literal[True]


class KernelSoftConditionReparam(pyro.infer.reparam.reparam.Reparam):
    """
    Reparametrizer that allows approximate soft conditioning on a :func:`pyro.deterministic`
    site using a kernel function that compares the observed and computed values,
    as in approximate Bayesian computation methods from classical statistics.

    This may be useful for estimating counterfactuals in Pyro programs
    corresponding to structural causal models with exogenous noise variables.

    The kernel function should return a score corresponding to the
    log-probability of the observed value given the computed value,
    which is then added to the model's unnormalized log-joint probability
    using :func:`pyro.factor`  :

        :math:`\\log p(v' | v) \\approx K(v, v')`

    The score tensor returned by the kernel function must have shape equal
    or broadcastable to the ``batch_shape`` of the site.

    .. note::
        Kernel functions must be positive-definite and symmetric.
        For example, :class:`~RBFKernel` returns a Normal log-probability
        of the distance between the observed and computed values.
    """

    def __init__(self, kernel: Kernel[torch.Tensor]):
        self.kernel = kernel
        super().__init__()

    def apply(
        self, msg: _DeterministicReparamMessage
    ) -> pyro.infer.reparam.reparam.ReparamResult:
        name = msg["name"]
        event_dim = msg["fn"].event_dim
        observed_value = msg["value"]
        computed_value = msg["fn"].base_dist.v

        if observed_value is not computed_value:  # fast path for trivial case
            approx_log_prob = self.kernel(computed_value, observed_value)
            pyro.factor(f"{name}_approx_log_prob", approx_log_prob)

        new_fn = pyro.distributions.Delta(observed_value, event_dim=event_dim).mask(
            False
        )
        return {"fn": new_fn, "value": observed_value, "is_observed": True}


class AutoSoftConditioning(pyro.infer.reparam.strategies.Strategy):
    """
    Automatic reparametrization strategy that allows approximate soft conditioning
    on ``pyro.deterministic`` sites in a Pyro model.

    This may be useful for estimating counterfactuals in Pyro programs corresponding
    to structural causal models with exogenous noise variables.

    This strategy uses :class:`~KernelSoftConditionReparam` to approximate
    the log-probability of the observed value given the computed value
    at each :func:`pyro.deterministic` site whose observed value is different
    from its computed value.

    .. note::
        Implementation details are subject to change.
        Currently uses a few pre-defined kernels such as :class:`~SoftEqKernel`
        and :class:`~RBFKernel` which are chosen for each site based on
        the site's ``event_dim`` and ``support``.
    """

    def __init__(self, *, scale: float = 1.0, alpha: float = 1.0):
        self.alpha = alpha
        self.scale = scale
        super().__init__()

    @staticmethod
    def site_is_deterministic(msg: pyro.infer.reparam.reparam.ReparamMessage) -> bool:
        return (
            msg["is_observed"]
            and isinstance(msg["fn"], pyro.distributions.MaskedDistribution)
            and isinstance(msg["fn"].base_dist, pyro.distributions.Delta)
        )

    def configure(
        self, msg: pyro.infer.reparam.reparam.ReparamMessage
    ) -> Optional[pyro.infer.reparam.reparam.Reparam]:
        if not self.site_is_deterministic(msg) or msg["value"] is msg["fn"].base_dist.v:
            return None

        if msg["fn"].base_dist.v.is_floating_point():
            scale = self.scale * functools.reduce(
                operator.mul, msg["fn"].event_shape, 1.0
            )
            return KernelSoftConditionReparam(
                RBFKernel(scale=scale, event_dim=len(msg["fn"].event_shape))
            )

        if msg["fn"].base_dist.v.dtype in (
            torch.bool,
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
        ):
            alpha = self.alpha * functools.reduce(
                operator.mul, msg["fn"].event_shape, 1.0
            )
            return KernelSoftConditionReparam(
                SoftEqKernel(alpha=alpha, event_dim=len(msg["fn"].event_shape))
            )

        raise NotImplementedError(
            f"Could not reparameterize deterministic site {msg['name']}"
        )
