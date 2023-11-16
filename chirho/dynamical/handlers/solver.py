from chirho.dynamical.internals.solver import Solver


class TorchDiffEq(Solver):
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
            **msg["kwargs"]
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
