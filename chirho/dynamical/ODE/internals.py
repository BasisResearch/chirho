import torch

from chirho.dynamical.internals import Dynamics, State


# noinspection PyPep8Naming
class ODEDynamics(Dynamics):
    def diff(self, dX: State[torch.Tensor], X: State[torch.Tensor]) -> None:
        raise NotImplementedError

    def observation(self, X: State[torch.Tensor]) -> None:
        raise NotImplementedError
