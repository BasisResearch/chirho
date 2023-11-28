import torch
from torch import tensor as tt, Tensor as TT
from collections import OrderedDict
import inspect
from .typedecs import KWType, KWTypeNNParams


def flatten_dparams(dparams: KWTypeNNParams) -> TT:
    return torch.cat([torch.flatten(dparams[k]) for k in dparams.keys()])


def unflatten_df_dparams(cat_df_dparams: TT, dparams: KWTypeNNParams) -> KWType:
    df_dparams = OrderedDict()

    last_fidx = 0
    for k in dparams.keys():
        slice_len = dparams[k].numel()
        cat_df_dparams_slice = cat_df_dparams[last_fidx:last_fidx + slice_len]
        last_fidx += slice_len

        df_dparams[k] = torch.unflatten(cat_df_dparams_slice, dim=0, sizes=dparams[k].shape)

    return df_dparams


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
