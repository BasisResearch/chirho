import contextlib
import math
import typing
import warnings
from typing import Any, Callable, Container, Dict, Generic, Optional, TypeVar

import pyro
import torch
from typing_extensions import ClassVar, ParamSpec

from chirho.indexed.handlers import DependentMaskMessenger, IndexPlatesMessenger
from chirho.indexed.ops import get_index_plates, indices_of
from chirho.observational.handlers.condition import Observations, condition
from chirho.observational.ops import Observation
from chirho.robust.internals.utils import (
    get_importance_traces,
    guess_max_plate_nesting,
    unbind_leftmost_dim,
)
from chirho.robust.ops import Point

pyro.settings.set(module_local_params=True)

P = ParamSpec("P")
Q = ParamSpec("Q")
S = TypeVar("S")
T = TypeVar("T")


class _UnmaskNamedSites(DependentMaskMessenger):
    names: Container[str]

    def __init__(self, names: Container[str]):
        self.names = names

    def get_mask(
        self,
        dist: pyro.distributions.Distribution,
        value: Optional[torch.Tensor],
        device: torch.device = torch.device("cpu"),
        name: Optional[str] = None,
    ) -> torch.Tensor:
        return torch.tensor(name is None or name in self.names, device=device)


class NMCLogPredictiveLikelihood(Generic[P, T], torch.nn.Module):
    model: Callable[P, Any]
    guide: Callable[P, Any]
    num_samples: int
    max_plate_nesting: Optional[int]

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
        self.max_plate_nesting = max_plate_nesting

    def forward(
        self, data: Point[T], *args: P.args, **kwargs: P.kwargs
    ) -> torch.Tensor:
        if self.max_plate_nesting is None:
            self.max_plate_nesting = guess_max_plate_nesting(
                self.model, self.guide, *args, **kwargs
            )
            warnings.warn(
                "Since max_plate_nesting is not specified, \
                the first call to NMCLogPredictiveLikelihood will not be seeded properly. \
                See https://github.com/BasisResearch/chirho/pull/408"
            )

        masked_guide = pyro.poutine.mask(mask=False)(self.guide)
        masked_model = _UnmaskNamedSites(names=set(data.keys()))(
            condition(data=data)(self.model)
        )
        log_weights = pyro.infer.importance.vectorized_importance_weights(
            masked_model,
            masked_guide,
            *args,
            num_samples=self.num_samples,
            max_plate_nesting=self.max_plate_nesting,
            **kwargs,
        )[0]
        return torch.logsumexp(log_weights, dim=0) - math.log(self.num_samples)


class BatchedObservations(Generic[T], Observations[T]):
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


class BatchedLatents(pyro.poutine.messenger.Messenger):
    num_particles: int
    name: str

    def __init__(self, num_particles: int, *, name: str = "__particles_mc"):
        assert num_particles > 0
        assert len(name) > 0
        self.num_particles = num_particles
        self.name = name
        super().__init__()

    def _pyro_sample(self, msg: dict) -> None:
        if self.num_particles > 1 and self.name not in indices_of(msg["fn"]):
            msg["fn"] = unbind_leftmost_dim(
                msg["fn"].expand((1,) + msg["fn"].batch_shape),
                self.name,
                size=self.num_particles,
            )


class PredictiveModel(Generic[P, T], torch.nn.Module):
    model: Callable[P, T]
    guide: Callable[P, Any]

    def __init__(
        self,
        model: torch.nn.Module,
        guide: torch.nn.Module,
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
                and not pyro.poutine.util.site_is_subsample(node)
            ]
        )

        with pyro.poutine.infer_config(lambda msg: {"_model_predictive_site": True}):
            with block_guide_sample_sites:
                with pyro.poutine.replay(trace=guide_tr.trace):
                    return self.model(*args, **kwargs)


class PredictiveFunctional(Generic[P, T], torch.nn.Module):
    model: Callable[P, Any]
    guide: Callable[P, Any]
    num_samples: int
    max_plate_nesting: Optional[int]

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
        self._predictive_model: PredictiveModel[P, T] = PredictiveModel(model, guide)
        self.num_samples = num_samples
        self.max_plate_nesting = max_plate_nesting

    def forward(self, *args: P.args, **kwargs: P.kwargs) -> Point[T]:
        if self.max_plate_nesting is None:
            self.max_plate_nesting = guess_max_plate_nesting(
                self.model, self.guide, *args, **kwargs
            )

        particles_plate = (
            contextlib.nullcontext()
            if self.num_samples == 1
            else pyro.plate(
                "__predictive_particles",
                self.num_samples,
                dim=-self.max_plate_nesting - 1,
            )
        )

        with pyro.poutine.trace() as model_tr, particles_plate:
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
    max_plate_nesting: int

    _data_plate_name: ClassVar[str] = "__particles_data"
    _mc_plate_name: ClassVar[str] = "__particles_mc"

    def __init__(
        self,
        model: torch.nn.Module,
        guide: torch.nn.Module,
        *,
        max_plate_nesting: int,
        num_samples: int = 1,
    ):
        super().__init__()
        self.model = model
        self.guide = guide
        self.max_plate_nesting = max_plate_nesting
        self.num_samples = num_samples

    def forward(
        self, data: Point[T], *args: P.args, **kwargs: P.kwargs
    ) -> torch.Tensor:
        get_nmc_traces = get_importance_traces(
            BatchedLatents(self.num_samples, name=self._mc_plate_name)(
                BatchedObservations(data, name=self._data_plate_name)(
                    PredictiveModel(self.model, self.guide)
                )
            ),
            pack=True,
        )

        with IndexPlatesMessenger(first_available_dim=-self.max_plate_nesting - 1):
            model_trace, guide_trace = get_nmc_traces(data, *args, **kwargs)
            plate_names = [
                get_index_plates()[p].name
                for p in [self._mc_plate_name, self._data_plate_name]
            ]

        wds = "".join(model_trace.plate_to_symbol[p] for p in plate_names)

        log_weights = typing.cast(torch.Tensor, 0.0)
        for site in model_trace.nodes.values():
            if site["type"] != "sample":
                continue
            log_weights += torch.einsum(
                site["packed"]["log_prob"]._pyro_dims + "->" + wds,
                [site["packed"]["log_prob"]],
            )

        for site in guide_trace.nodes.values():
            if site["type"] != "sample":
                continue
            log_weights -= torch.einsum(
                site["packed"]["log_prob"]._pyro_dims + "->" + wds,
                [site["packed"]["log_prob"]],
            )

        assert isinstance(log_weights, torch.Tensor)  # DEBUG
        assert len(log_weights.shape) == 2  # DEBUG
        assert log_weights.shape[0] == self.num_samples  # DEBUG
        return torch.logsumexp(log_weights, dim=0) - math.log(self.num_samples)
