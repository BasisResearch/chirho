import functools
from typing import TypeVar

import pyro
import pyro.distributions as dist
import torch

from chirho.counterfactual.handlers.selection import (
    SelectCounterfactual,
    SelectFactual,
    get_factual_indices,
)
from chirho.counterfactual.internals import no_ambiguity, site_is_ambiguous
from chirho.indexed.ops import gather, get_index_plates, indices_of, scatter
from chirho.observational.ops import observe

T = TypeVar("T")


class FactualConditioningMessenger(pyro.poutine.messenger.Messenger):
    """
    Effect handler for handling ambiguity in conditioning, for use with
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

    @_dispatched_observe.register(dist.FoldedDistribution)
    @_dispatched_observe.register(dist.Distribution)
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

    @_dispatched_observe.register
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
        obs_noise = gather(obs_noise, fw, event_dim=noise_event_dim).expand(
            obs_noise.shape
        )
        # obs_noise = pyro.sample(name + "_noise_prior", noise_dist, obs=obs_noise)
        obs_noise = observe(noise_dist, obs_noise, name=name + "_noise_prior")

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
        return pyro.sample(name, new_fn, obs=new_value)
