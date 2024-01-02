import collections
import math
import typing
from typing import Any, Callable, Generic, List, Optional, TypeVar

import pyro
import torch
from typing_extensions import ParamSpec

from chirho.indexed.handlers import IndexPlatesMessenger
from chirho.indexed.ops import get_index_plates, indices_of
from chirho.observational.handlers.condition import Observations
from chirho.robust.internals.utils import (
    bind_leftmost_dim,
    get_importance_traces,
    site_is_delta,
    unbind_leftmost_dim,
)
from chirho.robust.ops import Point

pyro.settings.set(module_local_params=True)

P = ParamSpec("P")
Q = ParamSpec("Q")
S = TypeVar("S")
T = TypeVar("T")


class BatchedLatents(pyro.poutine.messenger.Messenger):
    """
    Effect handler that adds a fresh batch dimension to all latent ``sample`` sites.
    Similar to wrapping a Pyro model in a ``pyro.plate`` context, but uses the machinery
    in ``chirho.indexed`` to automatically allocate and track the fresh batch dimension
    based on the ``name`` argument to ``BatchedLatents`` .

    .. warning:: Must be used in conjunction with :class:`~chirho.indexed.handlers.IndexPlatesMessenger` .

    :param int num_particles: Number of particles to use for parallelization.
    :param str name: Name of the fresh batch dimension.
    """

    num_particles: int
    name: str

    def __init__(self, num_particles: int, *, name: str = "__particles_mc"):
        assert num_particles > 0
        assert len(name) > 0
        self.num_particles = num_particles
        self.name = name
        super().__init__()

    def _pyro_sample(self, msg: dict) -> None:
        if (
            self.num_particles > 1
            and msg["value"] is None
            and not pyro.poutine.util.site_is_factor(msg)
            and not pyro.poutine.util.site_is_subsample(msg)
            and not site_is_delta(msg)
            and self.name not in indices_of(msg["fn"])
        ):
            msg["fn"] = unbind_leftmost_dim(
                msg["fn"].expand((1,) + msg["fn"].batch_shape),
                self.name,
                size=self.num_particles,
            )


class BatchedObservations(Generic[T], Observations[T]):
    """
    Effect handler that takes a dictionary of observation values for ``sample`` sites
    that are assumed to be batched along their leftmost dimension, adds a fresh named
    dimension using the machinery in ``chirho.indexed``, and reshapes the observation
    values so that the new ``chirho.observational.observe`` sites are batched along
    the fresh named dimension.

    Useful in combination with ``pyro.infer.Predictive`` which returns a dictionary
    of values whose leftmost dimension is a batch dimension over independent samples.

    .. warning:: Must be used in conjunction with :class:`~chirho.indexed.handlers.IndexPlatesMessenger` .

    :param Point[T] data: Dictionary of observation values.
    :param str name: Name of the fresh batch dimension.
    """

    name: str

    def __init__(self, data: Point[T], *, name: str = "__particles_data"):
        assert len(name) > 0
        self.name = name
        super().__init__(data)

    def _pyro_observe(self, msg: dict) -> None:
        super()._pyro_observe(msg)
        if msg["kwargs"]["name"] in self.data:
            rv, obs = msg["args"]
            event_dim = (
                len(rv.event_shape)
                if hasattr(rv, "event_shape")
                else msg["kwargs"].get("event_dim", 0)
            )
            batch_obs = unbind_leftmost_dim(obs, self.name, event_dim=event_dim)
            msg["args"] = (rv, batch_obs)


class PredictiveModel(Generic[P, T], torch.nn.Module):
    """
    Given a Pyro model and guide, constructs a new model that behaves as if
    the latent ``sample`` sites in the original model (i.e. the prior)
    were replaced by their counterparts in the guide (i.e. the posterior).

    .. note:: Sites that only appear in the model are annotated in traces
        produced by the predictive model with ``infer={"_model_predictive_site": True}`` .

    :param model: Pyro model.
    :param guide: Pyro guide.
    """

    model: Callable[P, T]
    guide: Callable[P, Any]

    def __init__(
        self,
        model: Callable[P, T],
        guide: Callable[P, Any],
    ):
        super().__init__()
        self.model = model
        self.guide = guide

    def forward(self, *args: P.args, **kwargs: P.kwargs) -> T:
        with pyro.poutine.trace() as guide_tr:
            self.guide(*args, **kwargs)

        block_guide_sample_sites = pyro.poutine.block(
            hide=[
                name
                for name, node in guide_tr.trace.nodes.items()
                if node["type"] == "sample"
            ]
        )

        with pyro.poutine.infer_config(
            config_fn=lambda msg: {"_model_predictive_site": True}
        ):
            with block_guide_sample_sites:
                with pyro.poutine.replay(trace=guide_tr.trace):
                    return self.model(*args, **kwargs)


class PredictiveFunctional(Generic[P, T], torch.nn.Module):
    """
    Functional that returns a batch of samples from the posterior predictive
    distribution of a Pyro model given a guide. As with ``pyro.infer.Predictive`` ,
    the returned values are batched along their leftmost positional dimension.

    Similar to ``pyro.infer.Predictive(model, guide, num_samples, parallel=True)``
    but uses :class:`~PredictiveModel` to construct the predictive distribution
    and infer the model ``sample`` sites whose values should be returned,
    and uses :class:`~BatchedLatents` to parallelize over samples from the guide.

    .. warning:: ``PredictiveFunctional`` currently applies its own internal instance of
        :class:`~chirho.indexed.handlers.IndexPlatesMessenger` ,
        so it may not behave as expected if used within another enclosing
        :class:`~chirho.indexed.handlers.IndexPlatesMessenger` context.

    :param model: Pyro model.
    :param guide: Pyro guide.
    :param num_samples: Number of samples to return.
    """

    model: Callable[P, Any]
    guide: Callable[P, Any]
    num_samples: int

    def __init__(
        self,
        model: torch.nn.Module,
        guide: torch.nn.Module,
        *,
        num_samples: int = 1,
        max_plate_nesting: Optional[int] = None,
        name: str = "__particles_predictive",
    ):
        super().__init__()
        self.model = model
        self.guide = guide
        self.num_samples = num_samples
        self._predictive_model: PredictiveModel[P, Any] = PredictiveModel(model, guide)
        self._first_available_dim = (
            -max_plate_nesting - 1 if max_plate_nesting is not None else None
        )
        self._mc_plate_name = name

    def forward(self, *args: P.args, **kwargs: P.kwargs) -> Point[T]:
        with IndexPlatesMessenger(first_available_dim=self._first_available_dim):
            with pyro.poutine.trace() as model_tr:
                with BatchedLatents(self.num_samples, name=self._mc_plate_name):
                    self._predictive_model(*args, **kwargs)

            return {
                name: bind_leftmost_dim(
                    node["value"],
                    self._mc_plate_name,
                    event_dim=len(node["fn"].event_shape),
                )
                for name, node in model_tr.trace.nodes.items()
                if node["type"] == "sample"
                and not pyro.poutine.util.site_is_subsample(node)
                and node["infer"].get("_model_predictive_site", False)
            }


class BatchedNMCLogPredictiveLikelihood(Generic[P, T], torch.nn.Module):
    model: Callable[P, Any]
    guide: Callable[P, Any]
    num_samples: int

    def __init__(
        self,
        model: torch.nn.Module,
        guide: torch.nn.Module,
        *,
        num_samples: int = 1,
        max_plate_nesting: Optional[int] = None,
        data_plate_name: str = "__particles_data",
        mc_plate_name: str = "__particles_mc",
    ):
        super().__init__()
        self.model = model
        self.guide = guide
        self.num_samples = num_samples
        self._first_available_dim = (
            -max_plate_nesting - 1 if max_plate_nesting is not None else None
        )
        self._data_plate_name = data_plate_name
        self._mc_plate_name = mc_plate_name

    def forward(
        self, data: Point[T], *args: P.args, **kwargs: P.kwargs
    ) -> torch.Tensor:
        get_nmc_traces = get_importance_traces(PredictiveModel(self.model, self.guide))

        with IndexPlatesMessenger(first_available_dim=self._first_available_dim):
            with BatchedLatents(self.num_samples, name=self._mc_plate_name):
                with BatchedObservations(data, name=self._data_plate_name):
                    model_trace, guide_trace = get_nmc_traces(*args, **kwargs)
            index_plates = get_index_plates()

        plate_name_to_dim = collections.OrderedDict(
            (index_plates[p].name, index_plates[p].dim)
            for p in [self._mc_plate_name, self._data_plate_name]
            if p in index_plates
        )

        log_weights = typing.cast(torch.Tensor, 0.0)
        for site in model_trace.nodes.values():
            if site["type"] != "sample":
                continue
            site_log_prob = site["log_prob"]
            for f in site["cond_indep_stack"]:
                if f.dim is not None and f.name not in plate_name_to_dim:
                    site_log_prob = site_log_prob.sum(f.dim, keepdim=True)
            log_weights = log_weights + site_log_prob

        for site in guide_trace.nodes.values():
            if site["type"] != "sample":
                continue
            site_log_prob = site["log_prob"]
            for f in site["cond_indep_stack"]:
                if f.dim is not None and f.name not in plate_name_to_dim:
                    site_log_prob = site_log_prob.sum(f.dim, keepdim=True)
            log_weights = log_weights - site_log_prob

        # sum out particle dimension and discard
        if self._mc_plate_name in index_plates:
            log_weights = torch.logsumexp(
                log_weights,
                dim=plate_name_to_dim[index_plates[self._mc_plate_name].name],
                keepdim=True,
            ) - math.log(self.num_samples)
            plate_name_to_dim.pop(index_plates[self._mc_plate_name].name)

        # permute if necessary to move plate dimensions to the left
        perm: List[int] = [dim for dim in plate_name_to_dim.values()]
        perm += [dim for dim in range(-len(log_weights.shape), 0) if dim not in perm]
        log_weights = torch.permute(log_weights, perm)

        # pack log_weights by squeezing out rightmost dimensions
        for _ in range(len(log_weights.shape) - len(plate_name_to_dim)):
            log_weights = log_weights.squeeze(-1)

        return log_weights
