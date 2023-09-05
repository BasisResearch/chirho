import torch

from chirho.dynamical import Dynamics
from chirho.dynamical.ops import State, simulate


# noinspection PyPep8Naming
class ODEDynamics(Dynamics):
    def diff(self, dX: State[torch.Tensor], X: State[torch.Tensor]):
        raise NotImplementedError

    def observation(self, X: State[torch.Tensor]):
        raise NotImplementedError

    def forward(self, initial_state: State[torch.Tensor], timespan, **kwargs):
        return simulate(self, initial_state, timespan, **kwargs)
