from typing import Any, Dict, Literal, Optional, TypedDict, TypeVar, Union

import pyro
import pyro.distributions as dist
import pyro.infer.reparam
import torch

from chirho.counterfactual.handlers.selection import (
    SelectCounterfactual,
    SelectFactual,
    get_factual_indices,
)
from chirho.counterfactual.internals import expand_obs_value_inplace_
from chirho.indexed.ops import gather, indices_of, scatter, union

T = TypeVar("T")


def site_is_ambiguous(msg: Dict[str, Any]) -> bool:
    """
    Helper function used with :func:`pyro.condition` to determine
    whether a site is observed or ambiguous.

    A sample site is ambiguous if it is marked observed, is downstream of an intervention,
    and the observed value's index variables are a strict subset of the distribution's
    indices and hence require clarification of which entries of the random variable
    are fixed/observed (as opposed to random/unobserved).
    """
    if not (
        msg["type"] == "sample"
        and msg["is_observed"]
        and not pyro.poutine.util.site_is_subsample(msg)
    ):
        return False

    try:
        return not msg["infer"]["_specified_conditioning"]
    except KeyError:
        value_indices = indices_of(msg["value"], event_dim=len(msg["fn"].event_shape))
        dist_indices = indices_of(msg["fn"])
        return (
            bool(union(value_indices, dist_indices)) and value_indices != dist_indices
        )


def no_ambiguity(msg: Dict[str, Any]) -> Dict[str, Any]:
    """
    Helper function used with :func:`pyro.poutine.infer_config` to inform
    :class:`AmbiguousConditioningReparam` that all ambiguity in the current
    context has been resolved.
    """
    return {"_specified_conditioning": True}


class AmbiguousConditioningReparam(pyro.infer.reparam.reparam.Reparam):
    """
    Abstract base class for reparameterizers that handle ambiguous conditioning.
    """

    pass


class AmbiguousConditioningStrategy(pyro.infer.reparam.strategies.Strategy):
    """
    Abstract base class for strategies that handle ambiguous conditioning.
    """

    pass


CondStrategy = Union[
    Dict[str, AmbiguousConditioningReparam], AmbiguousConditioningStrategy
]


class AmbiguousConditioningReparamMessenger(
    pyro.poutine.reparam_messenger.ReparamMessenger
):
    config: CondStrategy

    def _pyro_sample(self, msg: pyro.infer.reparam.reparam.ReparamMessage) -> None:
        if site_is_ambiguous(msg):
            expand_obs_value_inplace_(msg)
            msg["infer"]["_specified_conditioning"] = False
        super()._pyro_sample(msg)
        msg["infer"]["_specified_conditioning"] = True


class ConditionReparamMsg(TypedDict):
    fn: pyro.distributions.Distribution
    value: torch.Tensor
    is_observed: Literal[True]


class ConditionReparamArgMsg(ConditionReparamMsg):
    name: str


class FactualConditioningReparam(AmbiguousConditioningReparam):
    """
    Factual conditioning reparameterizer.

    This :class:`pyro.infer.reparam.reparam.Reparam` is used to resolve inherent
    semantic ambiguity in conditioning in the presence of interventions by
    splitting the observed value into a factual and counterfactual component,
    associating the observed value with the factual random variable,
    and sampling the counterfactual random variable from its prior.
    """

    @pyro.poutine.infer_config(config_fn=no_ambiguity)
    def apply(self, msg: ConditionReparamArgMsg) -> ConditionReparamMsg:
        with SelectFactual():
            fv = pyro.sample(msg["name"] + "_factual", msg["fn"], obs=msg["value"])

        with SelectCounterfactual():
            cv = pyro.sample(msg["name"] + "_counterfactual", msg["fn"])

        event_dim = len(msg["fn"].event_shape)
        fw_indices = get_factual_indices()
        new_value: torch.Tensor = scatter(
            fv, fw_indices, result=cv.clone(), event_dim=event_dim
        )
        new_fn = dist.Delta(new_value, event_dim=event_dim).mask(False)
        return {"fn": new_fn, "value": new_value, "is_observed": True}


class MinimalFactualConditioning(AmbiguousConditioningStrategy):
    """
    Reparameterization strategy for handling ambiguity in conditioning, for use with
    counterfactual semantics handlers such as :class:`MultiWorldCounterfactual` .

    :class:`MinimalFactualConditioning` applies :class:`FactualConditioningReparam`
    instances to all ambiguous sample sites in a model.

    .. note::

        A sample site is ambiguous if it is marked observed, is downstream of an intervention,
        and the observed value's index variables are a strict subset of the distribution's
        indices and hence require clarification of which entries of the random variable
        are fixed/observed (as opposed to random/unobserved).

    """

    def configure(
        self, msg: pyro.infer.reparam.reparam.ReparamMessage
    ) -> Optional[FactualConditioningReparam]:
        if not site_is_ambiguous(msg):
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
        name, fn, value = msg["name"], msg["fn"], msg["value"]

        tfm = (
            fn.transforms[-1]
            if len(fn.transforms) == 1
            else dist.transforms.ComposeTransformModule(fn.transforms)
        )
        noise_dist = fn.base_dist
        noise_event_dim = len(noise_dist.event_shape)
        obs_event_dim = len(fn.event_shape)

        # factual world
        with SelectFactual(), pyro.poutine.infer_config(config_fn=no_ambiguity):
            new_base_dist = dist.Delta(value, event_dim=obs_event_dim).mask(False)
            new_noise_dist = dist.TransformedDistribution(new_base_dist, tfm.inv)
            obs_noise = pyro.sample(
                name + "_noise_likelihood", new_noise_dist, obs=tfm.inv(value)
            )

        # depends on strategy and indices of noise_dist
        fw = get_factual_indices()
        obs_noise = gather(obs_noise, fw, event_dim=noise_event_dim).expand(
            obs_noise.shape
        )
        obs_noise = pyro.sample(name + "_noise_prior", noise_dist, obs=obs_noise)

        # counterfactual world
        with SelectCounterfactual(), pyro.poutine.infer_config(config_fn=no_ambiguity):
            cf_noise_dist = dist.Delta(obs_noise, event_dim=noise_event_dim).mask(False)
            cf_obs_dist = dist.TransformedDistribution(cf_noise_dist, tfm)
            cf_obs_value = pyro.sample(name + "_cf_obs", cf_obs_dist)

        # merge
        new_value = scatter(
            value, fw, result=cf_obs_value.clone(), event_dim=obs_event_dim
        )
        new_fn = dist.Delta(new_value, event_dim=obs_event_dim).mask(False)
        return {"fn": new_fn, "value": new_value, "is_observed": msg["is_observed"]}


class AutoFactualConditioning(MinimalFactualConditioning):
    """
    Reparameterization strategy for handling ambiguity in conditioning, for use with
    counterfactual semantics handlers such as :class:`MultiWorldCounterfactual` .

    When the distribution is a :class:`pyro.distributions.TransformedDistribution`,
    :class:`AutoFactualConditioning` automatically applies :class:`ConditionTransformReparam`
    to the site. Otherwise, it behaves like :class:`MinimalFactualConditioning` .

    .. note::

        This strategy is applied by default via :class:`MultiWorldCounterfactual`
        and :class:`TwinWorldCounterfactual` unless otherwise specified.

    .. note::

        A sample site is ambiguous if it is marked observed, is downstream of an intervention,
        and the observed value's index variables are a strict subset of the distribution's
        indices and hence require clarification of which entries of the random variable
        are fixed/observed (as opposed to random/unobserved).

    """

    def configure(
        self, msg: pyro.infer.reparam.reparam.ReparamMessage
    ) -> Optional[FactualConditioningReparam]:
        if not site_is_ambiguous(msg):
            return None

        fn = msg["fn"]
        while hasattr(fn, "base_dist"):
            if isinstance(fn, dist.FoldedDistribution):
                return FactualConditioningReparam()
            elif isinstance(fn, dist.TransformedDistribution):
                return ConditionTransformReparam()
            else:
                fn = fn.base_dist

        return super().configure(msg)
