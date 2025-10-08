import pyro
import pyro.distributions as dist
import torch
from pyro.distributions.transforms import AffineTransform
from pyro.infer.reparam.reparam import Reparam


class NormalReparam(Reparam):
    def __init__(
        self,
    ):
        super().__init__()

    def apply(self, msg):
        name = msg["name"]
        fn = msg["fn"]
        value = msg["value"]
        is_observed = msg["is_observed"]

        msg

        loc = fn.loc
        scale = fn.scale

        shape = value.shape if value is not None else fn.batch_shape + fn.event_shape

        event_dim = fn.event_dim

        with pyro.poutine.mask(mask=False):
            base_noise = pyro.sample(
                f"{name}_base_noise",
                dist.Normal(torch.zeros(shape), torch.ones(shape)),
            ).to(loc.device)

        if is_observed:
            new_value = value

        else:
            transform = AffineTransform(loc, scale, event_dim=event_dim)
            new_value = transform(base_noise)

        return {"fn": fn, "value": new_value, "is_observed": is_observed}
