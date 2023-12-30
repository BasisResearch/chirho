import collections
import math
import typing
from typing import Any, Callable, Generic, List, Optional, TypeVar

import pyro
import torch
from typing_extensions import ParamSpec

from chirho.indexed.handlers import IndexPlatesMessenger
from chirho.indexed.ops import get_index_plates
from chirho.robust.internals.utils import (
    BatchedLatents,
    BatchedObservations,
    get_importance_traces,
)
from chirho.robust.ops import Point

pyro.settings.set(module_local_params=True)

P = ParamSpec("P")
Q = ParamSpec("Q")
S = TypeVar("S")
T = TypeVar("T")


class PredictiveModel(Generic[P, T], torch.nn.Module):
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
    ):
        super().__init__()
        self.model = model
        self.guide = guide
        self.num_samples = num_samples
        self._predictive_model: PredictiveModel[P, Any] = PredictiveModel(model, guide)
        self._first_available_dim = (
            -max_plate_nesting - 1 if max_plate_nesting is not None else None
        )

    def forward(self, *args: P.args, **kwargs: P.kwargs) -> Point[T]:
        with IndexPlatesMessenger(first_available_dim=self._first_available_dim):
            with pyro.poutine.trace() as model_tr, BatchedLatents(self.num_samples):
                self._predictive_model(*args, **kwargs)

        return {
            name: node["value"]
            for name, node in model_tr.trace.nodes.items()
            if node["type"] == "sample"
            and not pyro.poutine.util.site_is_subsample(node)
            and node["infer"].get("_model_predictive_site", False)
        }


class BatchedNMCLogPredictiveLikelihood(Generic[P, T], torch.nn.Module):
    model: Callable[P, Any]
    guide: Callable[P, Any]
    num_samples: int

    _data_plate_name: str = "__particles_data"
    _mc_plate_name: str = "__particles_mc"

    def __init__(
        self,
        model: torch.nn.Module,
        guide: torch.nn.Module,
        *,
        num_samples: int = 1,
        max_plate_nesting: Optional[int] = None,
    ):
        super().__init__()
        self.model = model
        self.guide = guide
        self.num_samples = num_samples
        self._first_available_dim = (
            -max_plate_nesting - 1 if max_plate_nesting is not None else None
        )

    def forward(
        self, data: Point[T], *args: P.args, **kwargs: P.kwargs
    ) -> torch.Tensor:
        get_nmc_traces = get_importance_traces(
            BatchedLatents(self.num_samples, name=self._mc_plate_name)(
                BatchedObservations(data, name=self._data_plate_name)(
                    PredictiveModel(self.model, self.guide)
                )
            )
        )

        with IndexPlatesMessenger(first_available_dim=self._first_available_dim):
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
                if f.vectorized and f.name not in plate_name_to_dim:
                    site_log_prob = site_log_prob.sum(f.dim, keepdim=True)
            log_weights += site_log_prob

        for site in guide_trace.nodes.values():
            if site["type"] != "sample":
                continue
            site_log_prob = site["log_prob"]
            for f in site["cond_indep_stack"]:
                if f.vectorized and f.name not in plate_name_to_dim:
                    site_log_prob = site_log_prob.sum(f.dim, keepdim=True)
            log_weights -= site_log_prob

        # sum out particle dimension and discard
        if self.num_samples > 1 and self._mc_plate_name in index_plates:
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
