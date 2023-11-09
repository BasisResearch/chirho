from typing import Any, Dict, Optional

import pyro

from chirho.dynamical.internals.solver import (
    SolverRuntimeCheckHandler,
    get_solver,
    get_solver_runtime_check_handler,
)


class RuntimeCheckDynamics(pyro.poutine.messenger.Messenger):
    runtime_check_handler: Optional[SolverRuntimeCheckHandler] = None

    def _process_message(self, msg: Dict[str, Any]):
        if self.runtime_check_handler is not None:
            self.runtime_check_handler._process_message(msg)
        super()._process_message(msg)

    def _pyro_simulate(self, msg) -> None:
        # Check whether the dynamics satisfy the assumptions of the solver.
        if msg["kwargs"].get("solver", None) is not None:
            solver = msg["kwargs"]["solver"]
        else:
            solver = get_solver()

        self.runtime_check_handler = get_solver_runtime_check_handler(solver)

    def _pyro_post_simulate(self, msg) -> None:
        self.runtime_check_handler = None
