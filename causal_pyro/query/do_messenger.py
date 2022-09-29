from typing import Dict

import pyro

from causal_pyro.primitives import Intervention, intervene


class Do(pyro.poutine.messenger.Messenger):
    """
    Intervene on values in a probabilistic program.
    """

    def __init__(self, actions: Dict[str, Intervention]):
        self.actions = actions
        super().__init__()

    def _pyro_sample(self, msg):
        is_intervened: bool = msg.get("is_intervened", (msg["name"] in self.actions))
        msg["is_intervened"] = is_intervened

    def _pyro_post_sample(self, msg):
        if msg["name"] in self.actions:
            msg["value"] = intervene(msg["value"], self.actions[msg["name"]])


do = pyro.poutine.handlers._make_handler(Do)[1]