from typing import Dict

import torch
import pyro
from pyro.nn import PyroModule

from chirho.observational.handlers import condition
from chirho.interventional.handlers import do
from chirho.counterfactual.handlers import MultiWorldCounterfactual
from chirho.indexed.ops import IndexSet, gather


def average_treatment_effect(
    model: PyroModule,
    theta_hat: Dict[str, torch.tensor],
    n_monte_carlo: int,
) -> torch.Tensor:
    """Compute the average treatment effect of a model."""
    model_at_theta = condition(data=theta_hat)(model)
    with MultiWorldCounterfactual():
        with do(actions=dict(A=(torch.tensor(0.0), torch.tensor(1.0)))):
            Ys = model_at_theta(N=n_monte_carlo)
        Y0 = gather(Ys, IndexSet(A={1}), event_dim=0)
        Y1 = gather(Ys, IndexSet(A={2}), event_dim=0)
    return pyro.deterministic("ATE", (Y1 - Y0).mean(dim=-1, keepdim=True).squeeze())
