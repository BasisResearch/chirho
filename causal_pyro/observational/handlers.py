import functools
import operator
from typing import Callable, Generic, Optional, TypeVar, Union

import pyro
import torch

T = TypeVar("T")

Kernel = Callable[[T, T], torch.Tensor]


def site_is_deterministic(msg: dict) -> bool:
    return (
        msg["is_observed"]
        and isinstance(msg["fn"], pyro.distributions.MaskedDistribution)
        and isinstance(msg["fn"].base_dist, pyro.distributions.Delta)
    )


class SoftEqKernel(torch.nn.Module):
    event_dim: int
    alpha: torch.Tensor

    def __init__(self, alpha: Union[float, torch.Tensor] = 1.0, *, event_dim: int = 0):
        super().__init__()
        self.register_buffer("alpha", torch.as_tensor(alpha))
        self.event_dim = event_dim

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        eq = (x == y).to(dtype=self.alpha.dtype)
        return (
            pyro.distributions.Bernoulli(probs=self.alpha)
            .expand([1] * self.event_dim)
            .to_event(self.event_dim)
            .log_prob(eq)
        )


class RBFKernel(torch.nn.Module):
    event_dim: int
    scale: torch.Tensor

    def __init__(self, scale: Union[float, torch.Tensor] = 1.0, *, event_dim: int = 0):
        super().__init__()
        self.register_buffer("scale", torch.as_tensor(scale))
        self.event_dim = event_dim

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        r = x - y
        event_shape = r.shape[len(r.shape) - self.event_dim :]
        scale = self.scale / functools.reduce(operator.mul, event_shape, 1.)
        return (
            pyro.distributions.Normal(loc=0.0, scale=scale)
            .expand([1] * self.event_dim)
            .to_event(self.event_dim)
            .log_prob(x - y)
        )


class KernelSoftConditionReparam(Generic[T], pyro.infer.reparam.reparam.Reparam):
    """
    Reparametrizer that allows approximate conditioning on a ``pyro.deterministic`` site.
    """

    def __init__(self, kernel: Kernel[T]):
        self.kernel = kernel
        super().__init__()

    def apply(
        self, msg: pyro.infer.reparam.reparam.ReparamMessage
    ) -> pyro.infer.reparam.reparam.ReparamResult:
        assert site_is_deterministic(msg), "Can only reparametrize deterministic sites"

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
    Reparametrization strategy that allows approximate conditioning on ``pyro.deterministic`` sites.
    """

    def __init__(self, scale: float, alpha: float):
        self.alpha = alpha
        self.scale = scale
        super().__init__()

    def configure(self, msg: dict) -> Optional[pyro.infer.reparam.reparam.Reparam]:
        if not site_is_deterministic(msg) or msg["value"] is msg["fn"].base_dist.v:
            return None

        event_dim = len(msg["fn"].event_shape)

        if msg["fn"].base_dist.v.is_floating_point():
            return KernelSoftConditionReparam(
                RBFKernel(scale=self.scale, event_dim=event_dim)
            )

        if msg["fn"].base_dist.v.dtype in (
            torch.bool,
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
        ):
            return KernelSoftConditionReparam(
                SoftEqKernel(alpha=self.alpha, event_dim=event_dim)
            )

        raise NotImplementedError(
            f"Could not reparameterize deterministic site {msg['name']}"
        )
