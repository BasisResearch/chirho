import collections
import math
import typing
from typing import Any, Callable, Generic, Optional, Tuple, TypeVar

import pyro
import torch
from typing_extensions import ParamSpec

from chirho.indexed.handlers import IndexPlatesMessenger
from chirho.indexed.ops import get_index_plates
from chirho.observational.handlers.predictive import BatchedLatents, BatchedObservations
from chirho.robust.ops import Point

pyro.settings.set(module_local_params=True)

P = ParamSpec("P")
Q = ParamSpec("Q")
S = TypeVar("S")
T = TypeVar("T")


def get_importance_traces(
    model: Callable[P, Any],
    guide: Optional[Callable[P, Any]] = None,
) -> Callable[P, Tuple[pyro.poutine.Trace, pyro.poutine.Trace]]:
    """
    Thin functional wrapper around :func:`~pyro.infer.enum.get_importance_trace`
    that cleans up the original interface to avoid unnecessary arguments
    and efficiently supports using the prior in a model as a default guide.

    :param model: Model to run.
    :param guide: Guide to run. If ``None``, use the prior in ``model`` as a guide.
    :returns: A function that takes the same arguments as ``model`` and ``guide`` and returns
        a tuple of importance traces ``(model_trace, guide_trace)``.
    """

    def _fn(
        *args: P.args, **kwargs: P.kwargs
    ) -> Tuple[pyro.poutine.Trace, pyro.poutine.Trace]:
        if guide is not None:
            model_trace, guide_trace = pyro.infer.enum.get_importance_trace(
                "flat", math.inf, model, guide, args, kwargs
            )
            return model_trace, guide_trace
        else:  # use prior as default guide, but don't run model twice
            model_trace, _ = pyro.infer.enum.get_importance_trace(
                "flat", math.inf, model, lambda *_, **__: None, args, kwargs
            )

            guide_trace = model_trace.copy()
            for name, node in list(guide_trace.nodes.items()):
                if node["type"] != "sample":
                    del model_trace.nodes[name]
                elif pyro.poutine.util.site_is_factor(node) or node["is_observed"]:
                    del guide_trace.nodes[name]
            return model_trace, guide_trace

    return _fn


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
            log_weights = torch.transpose(
                log_weights[None], -len(log_weights.shape) - 1, plate_name_to_dim[name].dim
            )

        # pack log_weights by squeezing out rightmost dimensions
        for _ in range(len(log_weights.shape) - len(plate_name_to_dim)):
            log_weights = log_weights.squeeze(-1)

        return log_weights
