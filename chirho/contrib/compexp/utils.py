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


def soft_relu(x: TT, temp: TT):
    """
    An experimental temperature based relaxation of the relu that maintains a non-negative
    range.
    See https://www.desmos.com/calculator/vnjgtf4f6l for more details.
    """

    inflection_point = -torch.log(temp) / temp
    bounded_x = torch.where(x < inflection_point, x, tt(0.0))
    lt_part = torch.exp(temp * bounded_x)
    gt_part = x + (1./temp) * (torch.log(temp) + 1)
    y = torch.where(x < inflection_point, lt_part, gt_part)

    assert not torch.any(torch.isnan(y))

    return y
