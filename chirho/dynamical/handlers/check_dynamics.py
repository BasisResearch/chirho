import pyro

from chirho.dynamical.internals.solver import check_dynamics, get_solver


class RuntimeCheckDynamics(pyro.poutine.messenger.Messenger):
    def _pyro_simulate(self, msg) -> None:
        # Check whether the dynamics satisfy the assumptions of the solver.
        dynamics, state, start_time, _ = msg["args"]

        if msg["kwargs"].get("solver", None) is not None:
            solver = msg["kwargs"]["solver"]
        else:
            solver = get_solver()

        if not check_dynamics(solver, dynamics, state, start_time):
            raise ValueError("Dynamics are not valid for the solver.")
