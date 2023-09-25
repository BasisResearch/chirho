from __future__ import annotations

import pyro

from chirho.dynamical.ops import Backend


class BackendHandler(pyro.poutine.messenger.Messenger):
    def __init__(self, backend: Backend):
        self.backend = backend
        super().__init__()

    def _pyro_simulate(self, msg) -> None:
        # Overwrite the backend in the message with the one we're handling.
        msg["kwargs"]["backend"] = self.backend
