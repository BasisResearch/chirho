from typing import Any, Dict

import pyro

from causal_pyro.indexed.ops import indices_of, union


class EventDimMessenger(pyro.poutine.messenger.Messenger):
    reinterpreted_batch_ndim: int

    def __init__(self, reinterpreted_batch_ndim: int = 0):
        self.reinterpreted_batch_ndim = reinterpreted_batch_ndim
        super().__init__()

    def _pyro_sample(self, msg: dict) -> None:
        msg["fn"] = msg["fn"].to_event(self.reinterpreted_batch_ndim)


def site_is_ambiguous(msg: Dict[str, Any]) -> bool:
    """
    Helper function used with :func:`observe` to determine
    whether a site is observed or ambiguous.
    A sample site is ambiguous if it is marked observed, is downstream of an intervention,
    and the observed value's index variables are a strict subset of the distribution's
    indices and hence require clarification of which entries of the random variable
    are fixed/observed (as opposed to random/unobserved).
    """
    rv, obs = msg["args"][:2]
    value_indices = indices_of(obs, event_dim=len(rv.event_shape))
    dist_indices = indices_of(rv)
    return (
        bool(union(value_indices, dist_indices)) and value_indices != dist_indices
    ) or not msg["infer"].get("_specified_conditioning", True)


def no_ambiguity(msg: Dict[str, Any]) -> Dict[str, Any]:
    """
    Helper function used with :func:`pyro.poutine.infer_config` to inform
    :class:`FactualConditioningMessenger` that all ambiguity in the current
    context has been resolved.
    """
    return {"_specified_conditioning": True}
