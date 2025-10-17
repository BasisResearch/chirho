import pyro
from pyro.poutine.messenger import Messenger
from pyro.distributions.torch_distribution import TorchDistributionMixin
from torch.distributions import TransformedDistribution
import pyro.distributions as dist

EXOGENATE_META_KEY = "exogenate_meta"

# TODO check if the base predicate matches Uniform(0, 1) first, and if it does, sample uniform and push through icdf, no unwrapping needed
#  because transforms handle that internally when calling icdf on them.


class ExogenateNoiseMessenger(Messenger):

    def _pyro_sample(self, msg: dict) -> None:
        exogenate_meta = msg.get("infer", {}).get(EXOGENATE_META_KEY, None)
        if exogenate_meta is None or "base_dist_predicate" not in exogenate_meta:
            return
        
        # TODO Bail if observed for now. I think in most cases though where we would want to observe an exogenated site and do inference wrt
        #  the noise, we would want to observe a subset of the worlds after exogenation happens? In that case it might be useful, if possible,
        #  to push back through inverse transforms and the cdf to get to the noise space? More to think about here...also look at soft conditioning.
        if msg.get("is_observed", False):
            raise NotImplementedError("Exogenation of observed sites is not supported.")
        
        base_dist_predicate = exogenate_meta["base_dist_predicate"]
        original_fn = msg["fn"]
        name = msg["name"]
        
        # Handle Independent wrapper
        current_dist = original_fn
        if isinstance(current_dist, dist.Independent):
            event_dim = current_dist.reinterpreted_batch_ndims
            current_dist = current_dist.base_dist
        else:
            event_dim = original_fn.event_dim if hasattr(original_fn, 'event_dim') else 0
        
        # Unwrap TransformedDistribution layers and collect transforms until predicate matches
        transforms = []
        while isinstance(current_dist, TransformedDistribution) and not base_dist_predicate(current_dist):
            # Prepend transforms to maintain correct order when applying them later
            transforms = list(current_dist.transforms) + transforms
            current_dist = current_dist.base_dist
        
        # Validate that we found a matching distribution
        if not base_dist_predicate(current_dist):
            raise ValueError(
                f"Could not find base distribution matching predicate in distribution chain. "
                f"Reached {type(current_dist).__name__} instead."
            )
        
        # Validate that the distribution is Pyro-compatible
        if not isinstance(current_dist, TorchDistributionMixin):
            raise TypeError(
                f"Cannot exogenate to {type(current_dist).__name__}: not a Pyro-compatible distribution. "
                f"The distribution must be an instance of a Pyro distribution (have TorchDistributionMixin). "
                f"Consider exogenating to a base Pyro distribution like Normal, Exponential, etc. See Pyro's"
                f"LogNormal implementation for an example of how define a distribution who's base is a Pyro distribution."
            )
        
        # Sample noise from the base distribution
        u = pyro.sample(f"{name}_u", current_dist)  # TODO make _u suffix passable to the handler.
        
        # Apply transforms to push noise back to original space
        x = u
        for transform in transforms:
            x = transform(x)
        
        # Create a new Delta sample site that interventions can target
        # Store original_fn but omit base_dist_predicate key to prevent re-triggering
        delta_infer = {EXOGENATE_META_KEY: {"original_fn": original_fn}}
        msg["value"] = pyro.sample(name, dist.Delta(x, event_dim=event_dim), infer=delta_infer)
        # Stop so that everything else looks at the new sample site instead.
        msg["stop"] = True


def sample_exogenated(name, fn, base_dist_predicate, **kwargs):
    """
    Sample from a distribution while exogenating noise to a base distribution.
    
    This function unwraps TransformedDistribution layers until reaching a base distribution
    that matches the provided predicate. It samples noise from that base, then applies
    transforms to get back to the original distribution's space. This creates separate
    sample sites for noise and reparameterized value, allowing interventions that leave upstream
    noise shared and intact across parallel worlds.
    
    :param name: Name of the sample site
    :param fn: Distribution to sample from (may be transformed)
    :param base_dist_predicate: Callable that takes a distribution and returns True
                                 if it's the desired base to exogenate to.
                                 The matched distribution must be Pyro-compatible.
    :param kwargs: Additional keyword arguments to pass to pyro.sample
    :return: Sample value
    
    Example::
        # Unwrap to any Normal distribution in the chain
        sample_exogenated("x", dist.LogNormal(0, 1), lambda d: isinstance(d, dist.Normal))
        
        # Unwrap to Normal with specific parameters
        sample_exogenated("z", transformed_dist,
                         lambda d: isinstance(d, dist.Normal) and d.loc == 0)
    """
    infer = dict(kwargs.pop("infer", {}))
    infer[EXOGENATE_META_KEY] = {"base_dist_predicate": base_dist_predicate}
    return pyro.sample(name, fn, infer=infer, **kwargs)
