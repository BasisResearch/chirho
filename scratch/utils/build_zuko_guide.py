import pyro
import torch
import zuko
from pyro.contrib.zuko import ZukoToPyro
from pyro.contrib.easyguide import easy_guide


def build_zuko_guide(model, *args, **kwargs):
    # features=num_latents, context=0, transforms=1, hidden_features=(32, 32)
    flow = zuko.flows.NSF(*args, **kwargs)
    flow.transform = flow.transform.inv  # inverse autoregressive flow (IAF) are fast to sample from

    @easy_guide(model)
    def zuko_guide(self):
        # Match everything cz we're going to sample them as a joint normalizing flow.
        group = self.group(match=".*")

        return group.sample("joint_nf", ZukoToPyro(flow()))[-1]

    zuko_guide._parameters = dict(flow.named_parameters())

    return zuko_guide
