from typing import Any, Callable, Generic, Optional, TypeVar

import pyro
import torch
from typing_extensions import ParamSpec

from chirho.indexed.handlers import IndexPlatesMessenger
from chirho.robust.internals.nmc import BatchedLatents
from chirho.robust.internals.utils import bind_leftmost_dim
from chirho.robust.ops import Point

P = ParamSpec("P")
S = TypeVar("S")
T = TypeVar("T")


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
    when :class:`~chirho.robust.handlers.predictive.PredictiveModel` is used to construct
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
