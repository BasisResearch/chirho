from typing import Any, Callable, Generic, Mapping, Optional, TypeVar

import pyro
import torch
from typing_extensions import ParamSpec

from chirho.indexed.handlers import IndexPlatesMessenger
from chirho.indexed.ops import indices_of
from chirho.observational.handlers.condition import Observations
from chirho.observational.internals import (
    bind_leftmost_dim,
    site_is_delta,
    unbind_leftmost_dim,
)
from chirho.observational.ops import Observation

P = ParamSpec("P")
S = TypeVar("S")
T = TypeVar("T")

Point = Mapping[str, Observation[T]]


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
    guide: Optional[Callable[P, Any]]

    def __init__(
        self,
        model: Callable[P, T],
        guide: Optional[Callable[P, Any]] = None,
    ):
        super().__init__()
        self.model = model
        self.guide = guide

    def forward(self, *args: P.args, **kwargs: P.kwargs) -> T:
        """
        Returns a sample from the posterior predictive distribution.

        :return: Sample from the posterior predictive distribution.
        :rtype: T
        """
        with pyro.poutine.infer_config(
            config_fn=lambda msg: {"_model_predictive_site": False}
        ):
            with pyro.poutine.trace() as guide_tr:
                if self.guide is not None:
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
    Functional that returns a batch of samples from the predictive
    distribution of a Pyro model. As with ``pyro.infer.Predictive`` ,
    the returned values are batched along their leftmost positional dimension.

    Similar to ``pyro.infer.Predictive(model, guide, num_samples, parallel=True)``
    when :class:`~chirho.observational.handlers.predictive.PredictiveModel` is used to construct
    the ``model`` argument and infer the ``sample`` sites whose values should be returned,
    and uses :class:`~BatchedLatents` to parallelize over samples from the model.

    .. warning:: ``PredictiveFunctional`` currently applies its own internal instance of
        :class:`~chirho.indexed.handlers.IndexPlatesMessenger` ,
        so it may not behave as expected if used within another enclosing
        :class:`~chirho.indexed.handlers.IndexPlatesMessenger` context.

    :param model: Pyro model.
    :param num_samples: Number of samples to return.
    """

    model: Callable[P, Any]
    num_samples: int

    def __init__(
        self,
        model: torch.nn.Module,
        *,
        num_samples: int = 1,
        max_plate_nesting: Optional[int] = None,
        name: str = "__particles_predictive",
    ):
        super().__init__()
        self.model = model
        self.num_samples = num_samples
        self._first_available_dim = (
            -max_plate_nesting - 1 if max_plate_nesting is not None else None
        )
        self._mc_plate_name = name

    def forward(self, *args: P.args, **kwargs: P.kwargs) -> Point[T]:
        """
        Returns a batch of samples from the posterior predictive distribution.

        :return: Dictionary of samples from the posterior predictive distribution.
        :rtype: Point[T]
        """
        with IndexPlatesMessenger(first_available_dim=self._first_available_dim):
            with pyro.poutine.trace() as model_tr:
                with BatchedLatents(self.num_samples, name=self._mc_plate_name):
                    with pyro.poutine.infer_config(
                        config_fn=lambda msg: {
                            "_model_predictive_site": msg["infer"].get(
                                "_model_predictive_site", True
                            )
                        }
                    ):
                        self.model(*args, **kwargs)

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
