import pyro.infer.reparam
import torch


def expand_obs_value_inplace_(msg: pyro.infer.reparam.reparam.ReparamMessage) -> None:
    """
    Slightly gross workaround that mutates the msg in place
    to avoid triggering overzealous validation logic in
    :class:~`pyro.poutine.reparam.ReparamMessenger`
    that uses cheaper tensor shape and identity equality checks as
    a conservative proxy for an expensive tensor value equality check.
    (see https://github.com/pyro-ppl/pyro/blob/685c7adee65bbcdd6bd6c84c834a0a460f2224eb/pyro/poutine/reparam_messenger.py#L99)  # noqa: E501
    This workaround is correct because these reparameterizers do not change
    the observed entries, it just packs counterfactual values around them;
    the equality check being approximated by that logic would still pass.
    """
    msg["value"] = torch.as_tensor(msg["value"])
    msg["infer"]["orig_shape"] = msg["value"].shape
    _custom_init = getattr(msg["value"], "_pyro_custom_init", False)
    msg["value"] = msg["value"].expand(
        torch.broadcast_shapes(
            msg["fn"].batch_shape + msg["fn"].event_shape,
            msg["value"].shape,
        )
    )
    setattr(msg["value"], "_pyro_custom_init", _custom_init)
