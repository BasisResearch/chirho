from __future__ import annotations

from typing import TypeVar

import torch

from chirho.dynamical.internals.solver import Solver

S = TypeVar("S")
T = TypeVar("T")


class TorchDiffEq(Solver[torch.Tensor]):
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


# Maybe rename to something more explicit like:
# class LazilyCompilingDiffEqDotJL(Solver[torch.Tensor]):
class DiffEqDotJL(Solver[torch.Tensor]):

    def __init__(self):
        super().__init__()

        self.solve_kwargs = dict()
        self._lazily_compiled_solver = None
        self._dynamics_that_solver_was_compiled_with = None

    # def __enter__(self):
    #     super().__enter__()
    #     # Clear on entrance so re-use of an instantiated solver handler
    #     #  doesn't result in unexpected behavior.
    #     self._lazily_compiled_solver = None

    def _pyro_simulate_point(self, msg) -> None:
        from chirho.dynamical.internals.backends.diffeqdotjl import (
            diffeqdotjl_simulate_point,
        )

        dynamics, initial_state_and_params, start_time, end_time = msg["args"]
        msg["kwargs"].update(self.solve_kwargs)

        msg["value"] = diffeqdotjl_simulate_point(
            dynamics, initial_state_and_params, start_time, end_time, **msg["kwargs"]
        )
        msg["done"] = True

    def _pyro_simulate_trajectory(self, msg) -> None:
        from chirho.dynamical.internals.backends.diffeqdotjl import (
            diffeqdotjl_simulate_trajectory,
        )

        dynamics, initial_state_and_params, timespan = msg["args"]
        msg["kwargs"].update(self.solve_kwargs)

        msg["value"] = diffeqdotjl_simulate_trajectory(
            dynamics, initial_state_and_params, timespan, **msg["kwargs"]
        )
        msg["done"] = True

    def _pyro_simulate_to_interruption(self, msg) -> None:
        from chirho.dynamical.internals.backends.diffeqdotjl import (
            diffeqdotjl_simulate_to_interruption,
        )

        interruptions, dynamics, initial_state_and_params, start_time, end_time = msg["args"]
        msg["kwargs"].update(self.solve_kwargs)
        msg["value"] = diffeqdotjl_simulate_to_interruption(
            interruptions,
            dynamics,
            initial_state_and_params,
            start_time,
            end_time,
            **msg["kwargs"],
        )
        msg["done"] = True

    def _pyro_check_dynamics(self, msg) -> None:
        raise NotImplementedError

    # TODO g179du91 move to parent class as other solvers might also need to lazily compile?
    def _pyro__lazily_compile_problem(self, msg) -> None:

        dynamics, initial_state_ao_params, start_time, end_time = msg["args"]

        if self._lazily_compiled_solver is None:
            from chirho.dynamical.internals.backends.diffeqdotjl import diffeqdotjl_compile_problem

            msg["kwargs"].update(self.solve_kwargs)

            self._lazily_compiled_solver = diffeqdotjl_compile_problem(
                dynamics, initial_state_ao_params, start_time, end_time, **msg["kwargs"]
            )
            self._dynamics_that_solver_was_compiled_with = dynamics
        elif dynamics is not self._dynamics_that_solver_was_compiled_with:
            raise ValueError(
                "Lazily compiling a solver for a different dynamics than the one that was previously compiled."
            )

        msg["value"] = self._lazily_compiled_solver
        msg["done"] = True
