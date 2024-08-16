import pyro
import torch
from contextlib import nullcontext
from pyro.infer.autoguide import AutoDelta
from pyro.infer import config_enumerate


class SVIIterStruct:
    def __init__(self, svi_iter, guide, losses):
        self.svi_iter = svi_iter
        self.guide = guide
        self.losses = losses


def build_svi_iter(
        model,
        variational_family,
        *args,
        lr=1e-3,
        model_for_guide_init=None,
        block_latents=None,
        enumeration=False,
        detach_losses=True,
        **kwargs
):
    if model_for_guide_init is None:
        model_for_guide_init = model
    guide = variational_family(model_for_guide_init, *args, **kwargs)
    if enumeration:
        elbo = pyro.infer.TraceEnum_ELBO()(model, guide)
    else:
        elbo = pyro.infer.Trace_ELBO()(model, guide)
    elbo()  # initialize parameters.
    # Don't use elbo parameters generally, as we don't want to include
    #  parameters that are outside of the guide.
    optim = torch.optim.Adam(guide.parameters(), lr=lr)
    losses = []

    maybe_mle = nullcontext() if block_latents is None else pyro.poutine.block(hide=block_latents)

    if not isinstance(maybe_mle, type(nullcontext())):
        # Require that all latent (unobserved) sites are actually blocked.
        print(f"In MLE mode. Ignoring log prob contributions from sites:\n {'\n'.join(block_latents)}].")
        assert isinstance(guide, AutoDelta), "MLE only supported for AutoDelta guides."

    def svi_iter():
        optim.zero_grad()
        with maybe_mle:
            loss = elbo()
        loss.backward()
        optim.step()

        if detach_losses:
            losses.append(loss.detach().item())
        else:
            losses.append(loss)

    # return svi_iter, guide, losses
    return SVIIterStruct(svi_iter, guide, losses)
