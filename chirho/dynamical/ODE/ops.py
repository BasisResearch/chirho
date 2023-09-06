import torch

from chirho.dynamical.internals import State, Trajectory
from chirho.dynamical.ODE.internals import ODEDynamics


def ode_simulate(
    dynamics: ODEDynamics, initial_state: State[torch.Tensor], timespan, **kwargs
) -> Trajectory[torch.Tensor]:
    raise NotImplementedError(
        "Simulating from a `ODEDynamics` model requires specifying a simulation backend using a context manager.\n"
        "For example, to simulate using the `torchdiffeq` backend, call the model as follows:\n\n"
        "    with chirho.dynamical.ODE.backends.TorchDiffEq():\n"
        "        trajectory = model(initial_state, timespan)\n\n"
    )
