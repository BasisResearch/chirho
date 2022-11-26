from typing import Optional

import torch

import pyro


def _site_is_deterministic(msg: dict) -> bool:
    return (
        msg["type"] == "sample"
        and msg["is_observed"]
        and isinstance(msg["fn"], pyro.distributions.MaskedDistribution)
        and isinstance(msg["fn"].base_dist, pyro.distributions.Delta)
    )


class KernelSoftConditionReparam(pyro.infer.reparam.reparam.Reparam):
    """
    Reparametrizer that allows approximate conditioning on a ``pyro.deterministic`` site.
    """
    def __init__(self, kernel: pyro.contrib.gp.kernels.Kernel):
        self.kernel = kernel
        super().__init__()

    def apply(
        self, msg: pyro.infer.reparam.reparam.ReparamMessage
    ) -> pyro.infer.reparam.reparam.ReparamResult:
        assert _site_is_deterministic(msg), "Can only reparametrize deterministic sites"
        assert msg["fn"].event_dim == self.kernel.input_dim

        name = msg["name"]
        event_dim = msg["fn"].event_dim
        observed_value = msg["value"]
        computed_value = msg["fn"].base_dist.v

        approx_log_prob = self.kernel(computed_value, observed_value)

        pyro.factor(f"{name}_approx_log_prob", approx_log_prob)

        new_fn = pyro.distributions.Delta(observed_value, event_dim=event_dim).mask(False)
        return {"fn": new_fn, "value": observed_value, "is_observed": True}

    
class DiscreteSoftConditionReparam(pyro.infer.reparam.reparam.Reparam):
    """
    Reparametrizer that allows approximate conditioning on a ``pyro.deterministic`` site
    whose value is a discrete variable.
    """
    def __init__(self, alpha: float):
        self.alpha = alpha
        super().__init__()

    def apply(
        self, msg: pyro.infer.reparam.reparam.ReparamMessage
    ) -> pyro.infer.reparam.reparam.ReparamResult:
        assert _site_is_deterministic(msg), "Can only reparametrize deterministic sites"

        name = msg["name"]
        event_dim = msg["fn"].event_dim
        observed_value = msg["value"]
        computed_value = msg["fn"].base_dist.v

        if computed_value.dtype in (torch.bool, torch.int8, torch.int16, torch.int32, torch.int64):
            approx_log_prob = torch.where(computed_value == observed_value, self.alpha, 1 - self.alpha)
        else:
            raise NotImplementedError(f"Cannot handle site {name} of type {computed_value.dtype}")

        pyro.factor(f"{name}_approx_log_prob", approx_log_prob)

        new_fn = pyro.distributions.Delta(observed_value, event_dim=event_dim).mask(False)
        return {"fn": new_fn, "value": observed_value, "is_observed": True}


class AutoSoftConditioning(pyro.infer.reparam.strategies.Strategy):
    """
    Reparametrization strategy that allows approximate conditioning on ``pyro.deterministic`` sites.
    """
    def __init__(self, alpha: float, kernel_params: dict):
        self.alpha = alpha
        self.kernel_params = kernel_params
        super().__init__()

    def _configure_kernel(
        self, msg: pyro.infer.reparam.reparam.ReparamMessage
    ) -> pyro.contrib.gp.kernels.Kernel:
        return pyro.contrib.gp.kernels.RBF(input_dim=msg["fn"].event_dim, **self.kernel_params)

    def configure(self, msg: dict) -> Optional[pyro.infer.reparam.reparam.Reparam]:
        if not _site_is_deterministic(msg) and msg["value"] is not msg["fn"].base_dist.v:
            return None

        if msg["fn"].base_dist.v.is_floating_point():
            return KernelSoftConditionReparam(kernel=self._kernel_fn(msg))

        if msg["fn"].base_dist.v.dtype in (torch.bool, torch.int8, torch.int16, torch.int32, torch.int64):
            return DiscreteSoftConditionReparam(alpha=self.alpha)

        raise NotImplementedError(f"Could not reparameterize deterministic site {msg['name']}")
