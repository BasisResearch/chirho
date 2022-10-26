from typing import Literal, Optional, Tuple

import pyro
import torch

from .dispatched_strategy import DispatchedStrategy


class AutoSoftConditioning(DispatchedStrategy):
    def __init__(self, kernel: pyro.contrib.gp.kernels.Kernel, alpha: float = 1.0):
        self.kernel = kernel
        self.alpha = torch.as_tensor(alpha)
        super().__init__()


@AutoSoftConditioning.register
def _auto_soft_conditioning_delta(
    self, fn: pyro.distributions.Delta, value=None, is_observed=False
) -> Optional[Tuple[pyro.distributions.Delta, torch.Tensor, Literal[True]]]:

    if not is_observed or value is fn.v:
        return None

    if fn.v.is_floating_point():
        approx_log_prob = self.kernel(fn.v, value)
    else:
        approx_log_prob = torch.where(fn.v == value, self.alpha, 1 - self.alpha)

    pyro.factor("approx_log_prob", approx_log_prob)
    return pyro.distributions.Delta(v=value, event_dim=fn.event_dim), value, True
