import functools
from typing import Callable, Dict, ParamSpec, TypeVar

import torch
import pyro
from pyro.nn import PyroModule

from chirho.observational.handlers import condition
from chirho.interventional.handlers import do
from chirho.counterfactual.handlers import MultiWorldCounterfactual
from chirho.indexed.ops import IndexSet, gather

P = ParamSpec("P")
T = TypeVar("T")


def average_treatment_effect(
    model: PyroModule,
    theta_hat: Dict[str, torch.Tensor],
    n_monte_carlo: int = 10000,
) -> torch.Tensor:
    """Compute the average treatment effect of a model."""
    model_at_theta = condition(data=theta_hat)(model)
    with MultiWorldCounterfactual():
        with do(actions=dict(A=(torch.tensor(0.0), torch.tensor(1.0)))):
            Ys = model_at_theta(N=n_monte_carlo)
        Y0 = gather(Ys, IndexSet(A={1}), event_dim=0)
        Y1 = gather(Ys, IndexSet(A={2}), event_dim=0)
    return pyro.deterministic("ATE", (Y1 - Y0).mean(dim=-1, keepdim=True).squeeze())


Functional = Callable[[Callable[P, T], Dict[str, torch.Tensor], int], torch.Tensor]


def dice_correction(functional: Functional[P, T]) -> Functional[P, T]:
    @functools.wraps(functional)
    def _corrected_functional(
        model: Callable[P, T], theta_hat: Dict[str, torch.Tensor], n_monte_carlo: int
    ) -> torch.Tensor:
        with pyro.poutine.trace() as tr:
            value: torch.Tensor = functional(model, theta_hat, n_monte_carlo)

        tr.trace.compute_log_prob()
        logp = torch.zeros_like(value)
        for node in tr.trace.nodes.values():
            if node["type"] != "sample" or pyro.poutine.util.site_is_subsample(node):
                continue

            if not node["is_observed"] and not node["fn"].has_rsample:
                logp += node["log_prob"].sum()

        dice_factor = torch.exp(logp - logp.detach())
        return dice_factor * value

    return _corrected_functional
