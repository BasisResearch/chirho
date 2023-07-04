import functools
from typing import Any, Dict, TypeVar

import pyro
import pyro.distributions as dist
import torch

from causal_pyro.counterfactual.handlers.selection import (
    SelectCounterfactual,
    SelectFactual,
    get_factual_indices,
)
from causal_pyro.indexed.ops import gather, get_index_plates, indices_of, scatter, union
from causal_pyro.observational.ops import observe

T = TypeVar("T")


def site_is_ambiguous(msg: Dict[str, Any]) -> bool:
    """
    Helper function used with :func:`observe` to determine
    whether a site is observed or ambiguous.

    A sample site is ambiguous if it is marked observed, is downstream of an intervention,
    and the observed value's index variables are a strict subset of the distribution's
    indices and hence require clarification of which entries of the random variable
    are fixed/observed (as opposed to random/unobserved).
    """
    rv, obs = msg["args"][:2]
    value_indices = indices_of(obs, event_dim=len(rv.event_shape))
    dist_indices = indices_of(rv)
    return (
        bool(union(value_indices, dist_indices)) and value_indices != dist_indices
    ) or not msg["infer"].get("_specified_conditioning", True)


def no_ambiguity(msg: Dict[str, Any]) -> Dict[str, Any]:
    """
    Helper function used with :func:`pyro.poutine.infer_config` to inform
    :class:`FactualConditioningMessenger` that all ambiguity in the current
    context has been resolved.
    """
    return {"_specified_conditioning": True}


class FactualConditioningMessenger(pyro.poutine.messenger.Messenger):
    """
    Reparameterization strategy for handling ambiguity in conditioning, for use with
    counterfactual semantics handlers such as :class:`MultiWorldCounterfactual` .
    """

    def _pyro_post_sample(self, msg: dict) -> None:
        # expand latent values to include all index plates
        if not msg["is_observed"] and not pyro.poutine.util.site_is_subsample(msg):
            rv, value, event_dim = msg["fn"], msg["value"], len(msg["fn"].event_shape)
            index_plates = get_index_plates()

            new_shape = list(value.shape)
            for k in set(indices_of(rv)) - set(indices_of(value, event_dim=event_dim)):
                dim = index_plates[k].dim
                new_shape = [1] * ((event_dim - dim) - len(new_shape)) + new_shape
                new_shape[dim - event_dim] = rv.batch_shape[dim]

            msg["value"] = value.expand(tuple(new_shape))

    def _pyro_observe(self, msg: dict) -> None:
        if "name" not in msg["kwargs"]:
            msg["kwargs"]["name"] = msg["name"]

        if not site_is_ambiguous(msg):
            return

        msg["value"] = self._dispatched_observe(*msg["args"], name=msg["name"])
        msg["done"] = True
        msg["stop"] = True

    @functools.singledispatchmethod
    def _dispatched_observe(self, rv, obs: torch.Tensor, name: str) -> torch.Tensor:
        raise NotImplementedError


@FactualConditioningMessenger._dispatched_observe.register(dist.FoldedDistribution)
@FactualConditioningMessenger._dispatched_observe.register(dist.Distribution)
def _observe_dist(
    self, rv: dist.Distribution, obs: torch.Tensor, name: str
) -> torch.Tensor:
    with pyro.poutine.infer_config(config_fn=no_ambiguity):
        with SelectFactual():
            fv = pyro.sample(name + "_factual", rv, obs=obs)

        with SelectCounterfactual():
            cv = pyro.sample(name + "_counterfactual", rv)

        event_dim = len(rv.event_shape)
        fw_indices = get_factual_indices()
        new_value: torch.Tensor = scatter(
            fv, fw_indices, result=cv.clone(), event_dim=event_dim
        )
        new_rv = dist.Delta(new_value, event_dim=event_dim).mask(False)
        return pyro.sample(name, new_rv, obs=new_value)


@FactualConditioningMessenger._dispatched_observe.register
def _observe_tfmdist(
    self, rv: dist.TransformedDistribution, value: torch.Tensor, name: str
) -> torch.Tensor:
    tfm = (
        rv.transforms[-1]
        if len(rv.transforms) == 1
        else dist.transforms.ComposeTransformModule(rv.transforms)
    )
    noise_dist = rv.base_dist
    noise_event_dim = len(noise_dist.event_shape)
    obs_event_dim = len(rv.event_shape)

    # factual world
    with SelectFactual(), pyro.poutine.infer_config(config_fn=no_ambiguity):
        new_base_dist = dist.Delta(value, event_dim=obs_event_dim).mask(False)
        new_noise_dist = dist.TransformedDistribution(new_base_dist, tfm.inv)
        obs_noise = pyro.sample(
            name + "_noise_likelihood", new_noise_dist, obs=tfm.inv(value)
        )

    # depends on strategy and indices of noise_dist
    fw = get_factual_indices()
    obs_noise = gather(obs_noise, fw, event_dim=noise_event_dim).expand(obs_noise.shape)
    # obs_noise = pyro.sample(name + "_noise_prior", noise_dist, obs=obs_noise)
    obs_noise = observe(noise_dist, obs_noise, name=name + "_noise_prior")

    # counterfactual world
    with SelectCounterfactual(), pyro.poutine.infer_config(config_fn=no_ambiguity):
        cf_noise_dist = dist.Delta(obs_noise, event_dim=noise_event_dim).mask(False)
        cf_obs_dist = dist.TransformedDistribution(cf_noise_dist, tfm)
        cf_obs_value = pyro.sample(name + "_cf_obs", cf_obs_dist)

    # merge
    new_value = scatter(value, fw, result=cf_obs_value.clone(), event_dim=obs_event_dim)
    new_fn = dist.Delta(new_value, event_dim=obs_event_dim).mask(False)
    return pyro.sample(name, new_fn, obs=new_value)


# TODO use this to handle independent transformed distributions
# @FactualConditioningMessenger._dispatch_observe.register
# def _observe_indep(self, rv: dist.Independent, obs: torch.Tensor, name: str) -> torch.Tensor:
#     with EventDimMessenger(rv.reinterpreted_batch_ndims):
#         return self._dispatch_observe(rv.base_dist, obs, name)
