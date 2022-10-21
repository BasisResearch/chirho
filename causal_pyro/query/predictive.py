from typing import Container

import pyro

from causal_pyro.primitives import intervene


class PredictiveMessenger(pyro.poutine.messenger.Messenger):
    """
    Effect handler to create a joint distribution over observations and predictive samples.
    """

    def __init__(self, names: Container[str]):
        self.names = names
        super().__init__()

    def _pyro_sample(self, msg):
        if msg["name"] in self.names:
            msg["fn"] = intervene(msg["fn"], msg["fn"])
