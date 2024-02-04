import torch
from torch import tensor as tt, Tensor as TT
from collections import OrderedDict
import inspect
from .typedecs import KWType, KWTypeNNParams


def msg_args_kwargs_to_kwargs(msg):

    ba = inspect.signature(msg["fn"]).bind(*msg["args"], **msg["kwargs"])
    ba.apply_defaults()

    return ba.arguments


# Keyword arguments From Trace
def kft(trace, *whitelist: str) -> KWType:
    # Copy the ordereddict.
    new_trace = OrderedDict(trace.nodes.items())

    # If a whitelist is provided, only include the keys in the whitelist.
    if whitelist:
        new_trace = OrderedDict([(k, v) for k, v in new_trace.items() if k in whitelist])
    else:
        # Remove the _INPUT and _RETURN nodes, if present.
        new_trace.pop("_INPUT", None)
        new_trace.pop("_RETURN", None)

    return OrderedDict(zip(new_trace.keys(), [v["value"] for v in new_trace.values()]))
