import collections
import math
import typing
from typing import Any, Callable, Generic, Optional, TypeVar

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


class BatchedNMCLogMarginalLikelihood(Generic[P, T], torch.nn.Module):
    r"""
    Approximates the log marginal likelihood induced by ``model`` and ``guide``
    using importance sampling at an arbitrary batch of :math:`N`
    points :math:`\{x_n\}_{n=1}^N`.

    .. math::
        \log \left(\frac{1}{M} \sum_{m=1}^M \frac{p(x_n \mid \theta_m) p(\theta_m) )}{q_{\phi}(\theta_m)} \right),
        \quad \theta_m \sim q_{\phi}(\theta),

    where :math:`q_{\phi}(\theta)` is the guide, and :math:`p(x_n \mid \theta_m) p(\theta_m)`
    is the model joint density of the data and the latents sampled from the guide.

    :param model: Python callable containing Pyro primitives.
    :type model: torch.nn.Module
    :param guide: Python callable containing Pyro primitives.
        Must only contain continuous latent variables.
    :type guide: torch.nn.Module
    :param num_samples: Number of Monte Carlo draws :math:`M`
        used to approximate marginal distribution, defaults to 1
    :type num_samples: int, optional
    """

    model: Callable[P, Any]
    guide: Optional[Callable[P, Any]]
    num_samples: int

    def __init__(
        self,
        model: torch.nn.Module,
        guide: Optional[torch.nn.Module] = None,
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
        """
        Computes the log predictive likelihood of ``data`` given ``model`` and ``guide``.

        :param data: Dictionary of observations.
        :type data: Point[T]
        :return: Log marginal likelihood at each datapoint.
        :rtype: torch.Tensor
        """
        get_nmc_traces = get_importance_traces(self.model, self.guide)

        with IndexPlatesMessenger(first_available_dim=self._first_available_dim):
            with BatchedLatents(self.num_samples, name=self._mc_plate_name):
                with BatchedObservations(data, name=self._data_plate_name):
                    model_trace, guide_trace = get_nmc_traces(*args, **kwargs)
            index_plates = get_index_plates()

        plate_name_to_dim = collections.OrderedDict(
            (p, index_plates[p])
            for p in [self._mc_plate_name, self._data_plate_name]
            if p in index_plates
        )
        plate_frames = set(plate_name_to_dim.values())

        log_weights = typing.cast(torch.Tensor, 0.0)
        for site in model_trace.nodes.values():
            if site["type"] != "sample":
                continue
            site_log_prob = site["log_prob"]
            for f in site["cond_indep_stack"]:
                if f.dim is not None and f not in plate_frames:
                    site_log_prob = site_log_prob.sum(f.dim, keepdim=True)
            log_weights = log_weights + site_log_prob

        for site in guide_trace.nodes.values():
            if site["type"] != "sample":
                continue
            site_log_prob = site["log_prob"]
            for f in site["cond_indep_stack"]:
                if f.dim is not None and f not in plate_frames:
                    site_log_prob = site_log_prob.sum(f.dim, keepdim=True)
            log_weights = log_weights - site_log_prob

        # sum out particle dimension and discard
        if self._mc_plate_name in index_plates:
            log_weights = torch.logsumexp(
                log_weights,
                dim=plate_name_to_dim[self._mc_plate_name].dim,
                keepdim=True,
            ) - math.log(self.num_samples)
            plate_name_to_dim.pop(self._mc_plate_name)

        # move data plate dimension to the left
        for name in reversed(plate_name_to_dim.keys()):
            log_weights = bind_leftmost_dim(log_weights, name)

        # pack log_weights by squeezing out rightmost dimensions
        for _ in range(len(log_weights.shape) - len(plate_name_to_dim)):
            log_weights = log_weights.squeeze(-1)

        return log_weights
