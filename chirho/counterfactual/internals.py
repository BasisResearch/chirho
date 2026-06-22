from typing import Any

import pyro

from chirho.indexed.ops import indices_of, union


class SpecifiedConditioningInferDict(pyro.poutine.runtime.InferDict):
    _specified_conditioning: bool


def site_is_ambiguous(msg: dict[str, Any]) -> bool:
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
    return (bool(union(value_indices, dist_indices)) and value_indices != dist_indices) or not msg["infer"].get(
        "_specified_conditioning", True
    )


def no_ambiguity(msg: pyro.poutine.messenger.Message) -> SpecifiedConditioningInferDict:
    """
    Helper function used with :func:`pyro.poutine.infer_config` to inform
    :class:`FactualConditioningMessenger` that all ambiguity in the current
    context has been resolved.
    """
    return {"_specified_conditioning": True}
