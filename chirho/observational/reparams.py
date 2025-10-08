import pyro
import pyro.distributions as dist
from pyro.distributions.transforms import AffineTransform
from pyro.infer.reparam.reparam import Reparam

from chirho.indexed.ops import IndexSet, indices_of


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

        if isinstance(fn, dist.Independent):
            base = fn.base_dist
        elif isinstance(fn, dist.Normal):
            base = fn
        else:
            raise ValueError(f"NormalReparam only supports Normal or Independent(Normal), got {type(fn)}")

        loc = base.loc
        scale = base.scale
        event_dim = fn.event_dim

        if is_observed:
            value_indices = indices_of(value)
            if value_indices == IndexSet():
                return {"fn": fn, "value": value, "is_observed": is_observed}

            if value_indices != IndexSet():
                raise NotImplementedError("Partially observed Normal reparameterization is not implemented.")

        if not is_observed:
            base_noise = pyro.sample(f"{name}_base_noise", dist.Normal(0.0, 1.0)).to(loc.device)
            transform = AffineTransform(loc, scale, event_dim=event_dim)

            new_value = transform(base_noise)
            new_fn = dist.Delta(new_value, event_dim=event_dim)

            return {"fn": new_fn, "value": new_value, "is_observed": is_observed}
