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
    Default strategy for handling ambiguity in conditioning.
    """

    @expand_reparam_msg_value_inplace
    def configure(
        self, msg: pyro.infer.reparam.reparam.ReparamMessage
    ) -> Optional[FactualConditioningReparam]:
        if not msg["is_observed"] or pyro.poutine.util.site_is_subsample(msg):
            return None

        return FactualConditioningReparam()
