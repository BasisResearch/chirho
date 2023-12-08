from __future__ import annotations

from typing import TypeVar

import torch

from chirho.dynamical.internals.solver import Solver

S = TypeVar("S")
T = TypeVar("T")


class TorchDiffEq(Solver[torch.Tensor]):
    """
    A dynamical systems solver backend for ordinary differential equations using
    `torchdiffeq <https://github.com/rtqichen/torchdiffeq>`_.
    When used in conjunction with :func:`simulate <chirho.dynamical.ops.simulate>`, as below, this backend will take
    responsibility for simulating the dynamical system defined by the arguments
    to :func:`simulate <chirho.dynamical.ops.simulate>`

    .. code-block:: python

        with TorchDiffEq():
            simulate(dynamics, initial_state, start_time, end_time)

    Additional details on the arguments below can be found in the
    `torchdiffeq documentation <https://github.com/rtqichen/torchdiffeq#keyword-arguments>`_

    :param rtol: The relative tolerance for the solver.
    :type rtol: float

    :param atol: The absolute tolerance for the solver.
    :type atol: float

    :param method: The solver method to use.
    :type method: str

    :param options: Additional options to pass to the solver.
    :type options: dict

    """

    def __init__(self, rtol=1e-7, atol=1e-9, method=None, options=None):
        self.rtol = rtol
        self.atol = atol
        self.method = method
        self.options = options
        self.odeint_kwargs = {
            "rtol": rtol,
            "atol": atol,
            "method": method,
            "options": options,
        }
        super().__init__()

    def _pyro_simulate_point(self, msg) -> None:
        from chirho.dynamical.internals.backends.torchdiffeq import (
            torchdiffeq_simulate_point,
        )

        dynamics, initial_state, start_time, end_time = msg["args"]
        msg["kwargs"].update(self.odeint_kwargs)

        msg["value"] = torchdiffeq_simulate_point(
            dynamics, initial_state, start_time, end_time, **msg["kwargs"]
        )
        msg["done"] = True

    def _pyro_simulate_trajectory(self, msg) -> None:
        from chirho.dynamical.internals.backends.torchdiffeq import (
            torchdiffeq_simulate_trajectory,
        )

        dynamics, initial_state, timespan = msg["args"]
        msg["kwargs"].update(self.odeint_kwargs)

        msg["value"] = torchdiffeq_simulate_trajectory(
            dynamics, initial_state, timespan, **msg["kwargs"]
        )
        msg["done"] = True

    def _pyro_simulate_to_interruption(self, msg) -> None:
        from chirho.dynamical.internals.backends.torchdiffeq import (
            torchdiffeq_simulate_to_interruption,
        )

        interruptions, dynamics, initial_state, start_time, end_time = msg["args"]
        msg["kwargs"].update(self.odeint_kwargs)
        msg["value"] = torchdiffeq_simulate_to_interruption(
            interruptions,
            dynamics,
            initial_state,
            start_time,
            end_time,
            **msg["kwargs"],
        )
        msg["done"] = True

    def _pyro_check_dynamics(self, msg) -> None:
        from chirho.dynamical.internals.backends.torchdiffeq import (
            torchdiffeq_check_dynamics,
        )

        dynamics, initial_state, start_time, end_time = msg["args"]
        msg["kwargs"].update(self.odeint_kwargs)

        torchdiffeq_check_dynamics(
            dynamics, initial_state, start_time, end_time, **msg["kwargs"]
        )
        msg["done"] = True
