from typing import TYPE_CHECKING, Tuple

import torch

from chirho.dynamical.internals import State, Trajectory
from chirho.dynamical.ODE.internals import ODEDynamics

if TYPE_CHECKING:
    from chirho.dynamical.handlers import Interruption


def _ode_simulate(
    dynamics: ODEDynamics, initial_state: State[torch.Tensor], timespan, **kwargs
) -> Trajectory[torch.Tensor]:
    raise NotImplementedError(
        "Simulating from a `ODEDynamics` model requires specifying a simulation backend using a context manager.\n"
        "For example, to simulate using the `torchdiffeq` backend, call the model as follows:\n\n"
        "    with chirho.dynamical.ODE.backends.TorchDiffEq():\n"
        "        trajectory = simulate(model, initial_state, timespan)\n\n"
    )


def _ode_simulate_to_interruption(
    dynamics: ODEDynamics,
    start_state: State[torch.Tensor],
    timespan,  # The first element of timespan is assumed to be the starting time.
    **kwargs,
) -> Tuple[
    Trajectory[torch.Tensor],
    Tuple["Interruption", ...],
    torch.Tensor,
    State[torch.Tensor],
]:
    raise NotImplementedError(
        "Simulating from a `ODEDynamics` model requires specifying a simulation backend using a context manager.\n"
        "For example, to simulate using the `torchdiffeq` backend, call the model as follows:\n\n"
        "    with chirho.dynamical.ODE.backends.TorchDiffEq():\n"
        "        trajectory = simulate(model, initial_state, timespan)\n\n"
    )
