from typing import Optional, List, Union

import torch

import pyro
from pyro.contrib.gp.kernels import Kernel
from pyro.distributions import Delta, MaskedDistribution, TransformedDistribution, transforms
from pyro.infer.reparam.reparam import Reparam


class ConditionalComposeTransformModule(transforms.ConditionalTransformModule):
    def __init__(self, transforms: List[Union[transforms.TransformModule, transforms.ConditionalTransformModule]]):
        self.transforms = [
            transforms.ConstantConditionalTransform(t)
            if not isinstance(t, transforms.ConditionalTransform)
            else t
            for t in transforms
        ]
        super().__init__(event_dim=transforms[0].event_dim)

    def condition(self, context: torch.Tensor):
        return transforms.ComposeTransformModule([t.condition(context) for t in self.transforms])


class TransformInferReparam(Reparam):

    def apply(self, msg):
        fn = msg["fn"]
        value = msg["value"]
        is_observed = msg["is_observed"]

        assert isinstance(fn, TransformedDistribution)
        assert value is not None

        t = fn.transforms[0] if len(fn.transforms) == 1 else \
            transforms.ComposeTransform(fn.transforms)

        obs_base_dist = Delta(value, event_dim=fn.event_dim).expand(fn.batch_shape).mask(False)
        latent_base_dist = TransformedDistribution(obs_base_dist, [t.inv])
        new_obs_dist = TransformedDistribution(latent_base_dist, [t])
        return {"fn": new_obs_dist, "value": value, "is_observed": is_observed}


class KernelABCReparam(Reparam):
    """
    Reparametrizer that allows approximate conditioning on a ``pyro.deterministic`` site.
    """

    def __init__(self, kernel: Kernel):
        self.kernel = kernel
        super().__init__()

    def apply(self, msg):
        name = msg["name"]
        fn = msg["fn"]
        observed_value = msg["value"]
        is_observed = msg["is_observed"]

        # TODO fail gracefully if applied to the wrong site
        # TODO disambiguate names with multi world counterfacutals...
        assert is_observed
        assert isinstance(fn, MaskedDistribution)
        fn = fn.base_dist
        assert isinstance(fn, Delta)
        assert fn.event_dim == self.kernel.input_dim

        computed_value = fn.v
        pyro.factor(name + "_factor", self.kernel(computed_value, observed_value))

        new_fn = Delta(observed_value, event_dim=fn.event_dim).mask(False)
        return {"fn": new_fn, "value": observed_value, "is_observed": True}


class AutoSoftConditioning(pyro.infer.reparam.strategies.Strategy):
    @staticmethod
    def _is_deterministic(msg):
        return (
            msg["type"] == "sample"
            and msg["is_observed"]
            and isinstance(msg["fn"], MaskedDistribution)
            and isinstance(msg["fn"].base_dist, Delta)
            and msg["fn"]._mask is False
            and msg["infer"].get("is_deterministic", False)
        )

    def configure(self, msg: dict) -> Optional[Reparam]:
        if not msg["is_observed"]:
            return None

        if isinstance(msg["fn"], TransformedDistribution):
            return TransformInferReparam()

        if self._is_deterministic(msg):
            return KernelABCReparam(
                kernel=pyro.contrib.gp.kernels.RBF(input_dim=msg["fn"].event_dim)
            )

        return None
