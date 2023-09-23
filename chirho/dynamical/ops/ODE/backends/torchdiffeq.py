import torch

from chirho.dynamical.ops.ODE import ODEBackend


class TorchDiffEqBackend(ODEBackend[torch.Tensor]):
    def __init__(self, simulation_args={}):
        self.simulation_args = simulation_args
