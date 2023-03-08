from typing import Any, Dict, Literal, Optional, TypedDict, TypeVar

import pyro
import pyro.distributions as dist
import pyro.infer.reparam
import torch

from causal_pyro.counterfactual.internals import expand_reparam_msg_value_inplace
from causal_pyro.counterfactual.selection import SelectCounterfactual, SelectFactual
from causal_pyro.primitives import scatter

T = TypeVar("T")


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


class ConditionReparamMsg(TypedDict):
    fn: pyro.distributions.Distribution
    value: torch.Tensor
    is_observed: Literal[True]


class ConditionReparamArgMsg(ConditionReparamMsg):
    name: str


class FactualConditioningReparam(AmbiguousConditioningReparam):
    """
    Factual conditioning reparameterizer.

    This reparameterizer is used to resolve ambiguity in conditioning by
    splitting the observed value into a factual and counterfactual component,
    associating the observed value with the factual random variable,
    and sampling the counterfactual random variable from its prior.
    """

    @pyro.poutine.infer_config(config_fn=no_ambiguity)
    def apply(self, msg: ConditionReparamArgMsg) -> ConditionReparamMsg:
        with SelectFactual() as fw:
            fv = pyro.sample(msg["name"] + "_factual", msg["fn"], obs=msg["value"])

        with SelectCounterfactual() as cw:
            cv = pyro.sample(msg["name"] + "_counterfactual", msg["fn"])

        event_dim = len(msg["fn"].event_shape)
        new_value: torch.Tensor = scatter(
            {fw.indices: fv, cw.indices: cv}, event_dim=event_dim
        )
        new_fn = dist.Delta(new_value, event_dim=event_dim).mask(False)
        return {"fn": new_fn, "value": new_value, "is_observed": True}


class AutoFactualConditioning(AmbiguousConditioningStrategy):
    """
    Default strategy for handling ambiguity in conditioning, for use with
    counterfactual semantics handlers such as :class:`MultiWorldCounterfactual` .

    This strategy automatically applies :class:`FactualConditioningReparam` to
    all sites that are observed and are downstream of an intervention, provided
    that the observed value's index variables are a strict subset of the distribution's
    indices and hence require clarification of which entries of the random variable
    are fixed/observed (as opposed to random/unobserved).

    .. note::

        This strategy is applied by default via :class:`MultiWorldCounterfactual`
        and :class:`TwinWorldCounterfactual` unless otherwise specified.
    """

    @expand_reparam_msg_value_inplace
    def configure(
        self, msg: pyro.infer.reparam.reparam.ReparamMessage
    ) -> Optional[FactualConditioningReparam]:
        if not msg["is_observed"] or pyro.poutine.util.site_is_subsample(msg):
            return None

        return FactualConditioningReparam()
