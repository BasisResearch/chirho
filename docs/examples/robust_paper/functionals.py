from typing import Callable
import torch
import pyro
from chirho.counterfactual.handlers import MultiWorldCounterfactual
from chirho.indexed.ops import IndexSet, gather
from chirho.interventional.handlers import do

pyro.settings.set(module_local_params=True)


class ATEFunctional(torch.nn.Module):
    def __init__(
        self, model: Callable, *, treatment_name: str = "A", num_monte_carlo: int = 1000
    ):
        super().__init__()
        self.model = model
        self.num_monte_carlo = num_monte_carlo
        self.treatment_name = treatment_name

    def forward(self, *args, **kwargs) -> torch.Tensor:
        """
        Computes the average treatment effect (ATE) of the model. Assumes that the treatment
        is binary and that the model returns the target response.

        :return: average treatment effect estimated using Monte Carlo
        :rtype: torch.Tensor
        """
        with MultiWorldCounterfactual():
            with pyro.plate(
                "monte_carlo_functional", size=self.num_monte_carlo, dim=-2
            ):
                treatment_dict = {
                    self.treatment_name: (torch.tensor(0.0), torch.tensor(1.0))
                }
                with do(actions=treatment_dict):
                    Ys = self.model(*args, **kwargs)
                Y0 = gather(Ys, IndexSet(A={1}), event_dim=0)
                Y1 = gather(Ys, IndexSet(A={2}), event_dim=0)
        # TODO: if response is scalar, we do we need to average over dim=-1?
        ate = (Y1 - Y0).mean(dim=-2, keepdim=True).mean(dim=-1, keepdim=True).squeeze()
        return pyro.deterministic("ATE", ate)
