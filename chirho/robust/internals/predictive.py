import contextlib
import math
import warnings
from typing import (
    Any,
    Callable,
    Container,
    Dict,
    Generic,
    Mapping,
    Optional,
    Tuple,
    TypeVar,
)

import pyro
import torch
from typing_extensions import ParamSpec

from chirho.indexed.handlers import DependentMaskMessenger, IndexPlatesMessenger
from chirho.indexed.ops import get_index_plates, indices_of
from chirho.observational.handlers.condition import Observations, condition
from chirho.robust.internals.utils import guess_max_plate_nesting, unbind_leftmost_dim
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


class BatchedObservations(Observations[torch.Tensor]):
    data: Dict[str, torch.Tensor]
    name: str
    size: int

    def __init__(
        self,
        data: Mapping[str, torch.Tensor],
        *,
        name: str = "__particles_data",
    ):
        assert len(name) > 0
        self.name = name
        self.size = next(iter(data.values())).shape[0] if data else 1
        assert all(v.shape[0] == self.size for v in data.values())
        super().__init__({k: v for k, v in data.items()})

    def _pyro_sample(self, msg: dict) -> None:
        if msg["name"] in self.data and not msg["infer"].get("_do_not_observe", False):
            event_dim = len(msg["fn"].event_shape)
            old_datum = self.data[msg["name"]]
            try:
                self.data[msg["name"]] = unbind_leftmost_dim(
                    old_datum, self.name, size=self.size, event_dim=event_dim
                )
                return super()._pyro_sample(msg)
            finally:
                self.data[msg["name"]] = old_datum


class BatchedLatents(pyro.poutine.messenger.Messenger):
    name: str
    num_particles: int

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


def get_importance_traces(
    model: Callable[P, Any],
    guide: Optional[Callable[P, Any]] = None,
    pack: bool = True,
) -> Callable[P, Tuple[pyro.poutine.Trace, pyro.poutine.Trace]]:
    def _fn(
        *args: P.args, **kwargs: P.kwargs
    ) -> Tuple[pyro.poutine.Trace, pyro.poutine.Trace]:
        if guide is not None:
            model_trace, guide_trace = pyro.infer.enum.get_importance_trace(
                "flat", math.inf, model, guide, args, kwargs
            )
            if pack:
                guide_trace.pack_tensors()
                model_trace.pack_tensors(guide_trace.plate_to_symbol)
            return model_trace, guide_trace
        else:  # use the prior as a guide, but don't run model twice
            model_trace, _ = pyro.infer.enum.get_importance_trace(
                "flat", math.inf, model, lambda *_, **__: None, args, kwargs
            )
            if pack:
                model_trace.pack_tensors()

            guide_trace = model_trace.copy()
            for name, node in list(guide_trace.nodes.items()):
                if node["type"] != "sample":
                    del model_trace[name]
                elif pyro.poutine.util.site_is_factor(node) or node["is_observed"]:
                    del guide_trace[name]

            return model_trace, guide_trace

    return _fn


class BatchedNMCLogPredictiveLikelihood(Generic[P], torch.nn.Module):
    model: Callable[P, Any]
    guide: Callable[P, Any]
    num_samples: int
    max_plate_nesting: int
    data_plate_name: str
    mc_plate_name: str

    def __init__(
        self,
        model: torch.nn.Module,
        guide: torch.nn.Module,
        *,
        max_plate_nesting: int,
        num_samples: int = 1,
        data_plate_name: str = "__particles_data",
        mc_plate_name: str = "__particles_mc",
    ):
        super().__init__()
        self.model = model
        self.guide = guide
        self.max_plate_nesting = max_plate_nesting
        self.num_samples = num_samples
        self.data_plate_name = data_plate_name
        self.mc_plate_name = mc_plate_name

    def _batched_predictive_model(
        self, data: Mapping[str, torch.Tensor], *args: P.args, **kwargs: P.kwargs
    ):
        predictive_model: PredictiveModel[P, Any] = PredictiveModel(
            self.model, self.guide
        )
        with BatchedLatents(self.num_samples, name=self.mc_plate_name):
            with BatchedObservations(data, name=self.data_plate_name):
                return predictive_model(*args, **kwargs)

    def forward(
        self, data: Mapping[str, torch.Tensor], *args: P.args, **kwargs: P.kwargs
    ) -> torch.Tensor:
        get_nmc_traces = get_importance_traces(
            self._batched_predictive_model, pack=True
        )
        with IndexPlatesMessenger(first_available_dim=-self.max_plate_nesting - 1):
            model_trace, guide_trace = get_nmc_traces(data, *args, **kwargs)
            plate_names = [
                get_index_plates()[p].name
                for p in [self.mc_plate_name, self.data_plate_name]
            ]

        wds = "".join(model_trace.plate_to_symbol[p] for p in plate_names)

        log_weights = 0.0
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

        assert isinstance(log_weights, torch.Tensor)
        assert len(log_weights.shape) == 2 and log_weights.shape[0] == self.num_samples

        log_weights = torch.logsumexp(log_weights, dim=0) - math.log(self.num_samples)
        return log_weights
