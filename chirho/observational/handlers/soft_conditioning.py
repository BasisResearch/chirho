from __future__ import annotations

import functools
import operator
from typing import Callable, Literal, Optional, Protocol, TypedDict, TypeVar, Union

import pyro
import pyro.distributions.constraints as constraints
import torch

T = TypeVar("T")


Kernel = Callable[[T, T], torch.Tensor]


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
        eq = (x == y).to(dtype=self.alpha.dtype)
        return (
            pyro.distributions.Bernoulli(probs=self.alpha)
            .expand([1] * self.support.event_dim)
            .to_event(self.support.event_dim)
            .log_prob(eq)
        )


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
