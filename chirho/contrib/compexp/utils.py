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
def kft(trace) -> KWType:
    # Copy the ordereddict.
    new_trace = OrderedDict(trace.nodes.items())
    # Remove the _INPUT and _RETURN nodes.
    del new_trace["_INPUT"]
    del new_trace["_RETURN"]
    return OrderedDict(zip(new_trace.keys(), [v["value"] for v in new_trace.values()]))
