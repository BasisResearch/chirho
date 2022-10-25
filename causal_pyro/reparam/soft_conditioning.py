from typing import Literal, Optional, Tuple

import pyro
import torch

from .dispatched_strategy import DispatchedStrategy


class AutoSoftConditioning(DispatchedStrategy):
    def __init__(self, kernel: pyro.contrib.gp.kernels.Kernel):
        self.kernel = kernel
        super().__init__()


@AutoSoftConditioning.register
def _auto_soft_conditioning_delta(
    self, fn: pyro.distributions.Delta, value=None, is_observed=False, name=""
) -> Optional[Tuple[pyro.distributions.Delta, torch.Tensor, Literal[True]]]:
    if not is_observed or value is fn.v:
        return None
    pyro.factor(name + "_factor", self.kernel(fn.v, value))
    return pyro.distributions.Delta(v=value, event_dim=fn.event_dim), value, True
