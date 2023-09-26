from __future__ import annotations

import pyro

from chirho.dynamical.ops import Solver


class SolverHandler(pyro.poutine.messenger.Messenger):
    def __init__(self, solver: Solver):
        self.solver = solver
        super().__init__()

    def _pyro_simulate(self, msg) -> None:
        # Overwrite the solver in the message with the one we're handling.
        msg["kwargs"]["solver"] = self.solver
