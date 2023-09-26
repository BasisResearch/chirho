from __future__ import annotations

import pyro


class Solver(pyro.poutine.messenger.Messenger):
    def _pyro_simulate(self, msg) -> None:
        # Overwrite the solver in the message with the enclosing solver when used as a context manager.
        msg["kwargs"]["solver"] = self
