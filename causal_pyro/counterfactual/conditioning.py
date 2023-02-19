import functools
from typing import Any, Callable, Dict, Literal, Optional, TypedDict, TypeVar

import pyro
import pyro.distributions as dist
import pyro.infer.reparam
import torch

from causal_pyro.counterfactual.selection import SelectCounterfactual, SelectFactual
from causal_pyro.primitives import gather, indices_of, merge

T = TypeVar("T")


def no_ambiguity(msg: Dict[str, Any]) -> Dict[str, Any]:
    return {"_specified_conditioning": True}


class AmbiguousConditioningReparam(pyro.infer.reparam.reparam.Reparam):
    pass


class AmbiguousConditioningStrategy(pyro.infer.reparam.strategies.Strategy):
    def __init_subclass__(cls, **kwargs) -> None:
        # TODO check to avoid wrapping configure more than once
        cls.configure = cls._expand_msg_value(cls.configure)
        return super().__init_subclass__(**kwargs)

    @staticmethod
    def _expand_msg_value(
        config_fn: Callable[..., Optional[AmbiguousConditioningReparam]]
    ) -> Callable[..., Optional[AmbiguousConditioningReparam]]:

        @functools.wraps(config_fn)
        def _wrapper(*args) -> Optional[AmbiguousConditioningReparam]:
            msg: pyro.infer.reparam.reparam.ReparamMessage = args[-1]

            if msg["infer"].get("_specified_conditioning", False):
                # avoid infinite recursion
                return None

            if (
                msg["is_observed"]
                and msg["value"] is not None
                and not pyro.poutine.util.site_is_subsample(msg)
            ):
                # XXX slightly gross workaround that mutates the msg in place to avoid
                #   triggering overzealous validation logic in pyro.poutine.reparam
                #   that uses cheaper tensor shape and identity equality checks as
                #   a conservative proxy for an expensive tensor value equality check.
                #   (see https://github.com/pyro-ppl/pyro/blob/685c7adee65bbcdd6bd6c84c834a0a460f2224eb/pyro/poutine/reparam_messenger.py#L99)  # noqa: E501
                #   This workaround is correct because these reparameterizers do not change
                #   the observed entries, it just packs counterfactual values around them;
                #   the equality check being approximated by that logic would still pass.
                msg["infer"]["orig_shape"] = msg["value"].shape
                _custom_init = getattr(msg["value"], "_pyro_custom_init", False)
                msg["value"] = msg["value"].expand(
                    torch.broadcast_shapes(
                        msg["fn"].batch_shape + msg["fn"].event_shape,
                        msg["value"].shape,
                    )
                )
                setattr(msg["value"], "_pyro_custom_init", _custom_init)

            return config_fn(*(args[:-1] + (msg,)))

        return _wrapper


class ConditionReparamMsg(TypedDict):
    fn: pyro.distributions.Distribution
    value: torch.Tensor
    is_observed: Literal[True]


class ConditionReparamArgMsg(ConditionReparamMsg):
    name: str


class FactualConditioningReparam(AmbiguousConditioningReparam):
    """
    Factual conditioning reparameterizer.
    """

    @pyro.poutine.infer_config(config_fn=no_ambiguity)
    def apply(self, msg: ConditionReparamArgMsg) -> ConditionReparamMsg:
        with SelectFactual() as fw:
            fv = pyro.sample(msg["name"] + "_factual", msg["fn"], obs=msg["value"])

        with SelectCounterfactual() as cw:
            cv = pyro.sample(msg["name"] + "_counterfactual", msg["fn"])

        event_dim = len(msg["fn"].event_shape)
        new_value: torch.Tensor = merge({fw.indices: fv, cw.indices: cv}, event_dim=event_dim)
        new_fn = dist.Delta(new_value, event_dim=event_dim).mask(False)
        return {"fn": new_fn, "value": new_value, "is_observed": True}


class MinimalFactualConditioning(AmbiguousConditioningStrategy):
    """
    Default strategy for handling ambiguity in conditioning.
    """

    def configure(
        self, msg: pyro.infer.reparam.reparam.ReparamMessage
    ) -> Optional[FactualConditioningReparam]:
        if not msg["is_observed"] or pyro.poutine.util.site_is_subsample(msg):
            return None

        return FactualConditioningReparam()


class ConditionTransformReparamMsg(TypedDict):
    fn: pyro.distributions.TransformedDistribution
    value: torch.Tensor
    is_observed: Literal[True]


class ConditionTransformReparamArgMsg(ConditionTransformReparamMsg):
    name: str


class ConditionTransformReparam(AmbiguousConditioningReparam):
    def apply(
        self, msg: ConditionTransformReparamArgMsg
    ) -> ConditionTransformReparamMsg:
        name, fn, value, is_observed = (
            msg["name"],
            msg["fn"],
            msg["value"],
            msg["is_observed"],
        )

        tfm = (
            fn.transforms[-1]
            if len(fn.transforms) == 1
            else dist.transforms.ComposeTransformModule(fn.transforms)
        )
        noise_dist = fn.base_dist
        noise_event_dim, obs_event_dim = len(noise_dist.event_shape), len(
            fn.event_shape
        )

        # factual world
        with SelectFactual() as fw, pyro.poutine.infer_config(config_fn=no_ambiguity):
            new_base_dist = dist.Delta(value, event_dim=obs_event_dim).mask(False)
            new_noise_dist = dist.TransformedDistribution(new_base_dist, tfm.inv)
            obs_noise = pyro.sample(name + "_noise_likelihood", new_noise_dist, obs=tfm.inv(value))

        # depends on strategy and indices of noise_dist
        obs_noise = gather(obs_noise, fw.indices, event_dim=noise_event_dim).expand(obs_noise.shape)
        # obs_noise = pyro.sample(name + "_noise_prior", noise_dist, obs=obs_noise)  # DEBUG

        # counterfactual world
        with SelectCounterfactual() as cw, pyro.poutine.infer_config(
            config_fn=no_ambiguity
        ):
            cf_noise_dist = dist.Delta(obs_noise, event_dim=noise_event_dim).mask(False)
            cf_obs_dist = dist.TransformedDistribution(cf_noise_dist, tfm)
            cf_obs_value = pyro.sample(name + "_cf_obs", cf_obs_dist)

        # merge
        new_value = merge(
            {fw.indices: value, cw.indices: cf_obs_value}, event_dim=obs_event_dim
        )
        new_fn = dist.Delta(new_value, event_dim=obs_event_dim).mask(False)
        return {"fn": new_fn, "value": new_value, "is_observed": is_observed}


class AutoFactualConditioning(MinimalFactualConditioning):
    """
    Strategy for handling ambiguity in conditioning.
    """

    def configure(
        self, msg: pyro.infer.reparam.reparam.ReparamMessage
    ) -> Optional[FactualConditioningReparam]:
        if not msg["is_observed"] or pyro.poutine.util.site_is_subsample(msg):
            return None

        fn = msg["fn"]
        while hasattr(fn, "base_dist"):
            if isinstance(fn, dist.TransformedDistribution):
                return ConditionTransformReparam()
            else:
                fn = fn.base_dist

        return super().configure(msg)
