from typing import Dict

import pyro

from causal_pyro.primitives import Intervention, intervene


class DoMessenger(pyro.poutine.messenger.Messenger):
    """
    Intervene on values in a probabilistic program.
    """

    def __init__(self, actions: Dict[str, Intervention]):
        self.actions = actions
        super().__init__()

    def _pyro_post_sample(self, msg):
        if msg["name"] in self.actions and not msg.get("no_intervene", False):
            msg["value"] = intervene(
                msg["value"],
                self.actions[msg["name"]],
                event_dim=len(msg["fn"].event_shape),
            )


do = pyro.poutine.handlers._make_handler(DoMessenger)[1]
