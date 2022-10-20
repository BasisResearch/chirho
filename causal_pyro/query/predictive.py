from typing import Dict, Optional, Union

import pyro

from causal_pyro.primitives import Intervention, intervene


class PredictiveMessenger(pyro.poutine.messenger.Messenger):
    """
    Effect handler to create a joint distribution over observations and predictive samples.
    """
    def _pyro_sample(self, msg):
        if msg["is_observed"] and not pyro.poutine.util.site_is_subsample(msg):
            msg["stop"] = True
            msg["done"] = True
            duplicate_fn = intervene(msg["fn"], msg["fn"])
            # TODO this does not commute with intervention queries
            with pyro.poutine.messenger.block_messengers(lambda m: m is self):
                msg["value"] = pyro.sample(
                    msg["name"],
                    duplicate_fn,
                    obs=msg["value"],
                    infer=msg["infer"].copy()
                )