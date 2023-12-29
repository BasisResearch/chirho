import contextlib
import math
import warnings
from typing import (
    Any,
    Callable,
    Container,
    Dict,
    Generic,
    List,
    Mapping,
    Optional,
    Tuple,
    TypeVar,
)

import pyro
import torch
from typing_extensions import Concatenate, ParamSpec

from chirho.indexed.handlers import DependentMaskMessenger
from chirho.observational.handlers.condition import Observations, condition
from chirho.robust.internals.utils import guess_max_plate_nesting
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
        self._predictive_model = PredictiveModel(model, guide)
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


class BatchedObservations(Observations[torch.Tensor]):
    data: Dict[str, torch.Tensor]
    name: str
    size: int
    dim: int
    plate: pyro.poutine.indep_messenger.IndepMessenger

    def __init__(
        self,
        data: Mapping[str, torch.Tensor],
        dim: int,
        *,
        name: str = "__particles_data",
    ):
        self.dim = dim

        assert len(name) > 0
        self.name = name

        self.size = next(iter(data.values())).shape[0] if data else 1

        assert all(v.shape[0] == self.size for v in data.values())
        super().__init__({k: v for k, v in data.items()})

        self.plate = pyro.plate(name=self.name, size=self.size, dim=self.dim)

    def _pyro_sample(self, msg: dict) -> None:
        if msg["name"] not in self.data:
            return super()._pyro_sample(msg)

        old_datum, event_dim = self.data[msg["name"]], len(msg["fn"].event_shape)

        try:
            new_datum: torch.Tensor = torch.as_tensor(old_datum)
            with self.plate:  # enter plate context here to ensure plate.dim is set
                while self.plate.dim - event_dim < -len(new_datum.shape):
                    new_datum = new_datum[None]
                if new_datum.shape[0] == 1 and old_datum.shape[0] != 1:
                    new_datum = torch.transpose(
                        new_datum, -len(old_datum.shape), self.plate.dim - event_dim
                    )
                self.data[msg["name"]] = new_datum
                return super()._pyro_sample(msg)
        finally:
            self.data[msg["name"]] = old_datum


def get_importance_traces(
    model: Callable[P, Any],
    guide: Optional[Callable[P, Any]] = None,
    max_plate_nesting: Optional[int] = None,
    pack: bool = True,
) -> Callable[P, Tuple[pyro.poutine.Trace, pyro.poutine.Trace]]:
    def _fn(
        *args: P.args, **kwargs: P.kwargs
    ) -> Tuple[pyro.poutine.Trace, pyro.poutine.Trace]:
        if guide is not None:
            model_trace, guide_trace = pyro.infer.enum.get_importance_trace(
                "flat", max_plate_nesting, model, guide, args, kwargs
            )
            if pack:
                guide_trace.pack_tensors()
                model_trace.pack_tensors(guide_trace.plate_to_symbol)
            return model_trace, guide_trace
        else:
            model_trace, _ = pyro.infer.enum.get_importance_trace(
                "flat", max_plate_nesting, model, lambda *_, **__: None, args, kwargs
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


def BatchedNMCLogPredictiveLikelihood(
    model: Callable[P, Any],
    guide: Callable[P, Any],
    *,
    max_plate_nesting: int,
    num_samples: int = 1,
    data_plate_name: str = "__particles_data",
    mc_plate_name: str = "__particles_mc",
) -> Callable[Concatenate[Mapping[str, torch.Tensor], P], torch.Tensor]:

    predictive_model = PredictiveModel(model, guide)

    def batched_predictive_model(
        data: Mapping[str, torch.Tensor], *args: P.args, **kwargs: P.kwargs
    ):
        with pyro.plate(mc_plate_name, num_samples, dim=-max_plate_nesting - 2):
            with BatchedObservations(
                data, dim=-max_plate_nesting - 1, name=data_plate_name
            ):
                return predictive_model(*args, **kwargs)

    get_nmc_predictive_traces = get_importance_traces(
        batched_predictive_model,
        guide=None,
        max_plate_nesting=max_plate_nesting + 2,
        pack=True,
    )

    plate_names: List[str] = [mc_plate_name, data_plate_name]

    def _fn(
        data: Mapping[str, torch.Tensor], *args: P.args, **kwargs: P.kwargs
    ) -> torch.Tensor:
        model_trace, guide_trace = get_nmc_predictive_traces(data, *args, **kwargs)

        wds = "".join(model_trace.plate_to_symbol[p] for p in plate_names)

        log_weights = torch.as_tensor(0.0, device=next(iter(data.values())).device)
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

        assert len(log_weights.shape) == 2 and log_weights.shape[0] == num_samples

        log_weights = torch.logsumexp(log_weights, dim=0) - math.log(num_samples)
        return log_weights

    return _fn
