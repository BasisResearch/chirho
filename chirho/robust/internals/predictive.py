import contextlib
import math
import warnings
from typing import Any, Callable, Container, Dict, Generic, List, Optional, TypeVar

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

        with pyro.poutine.trace() as guide_tr, particles_plate:
            self.guide(*args, **kwargs)

        block_guide_sample_sites = pyro.poutine.block(
            hide=[
                name
                for name, node in guide_tr.trace.nodes.items()
                if node["type"] == "sample"
                and not pyro.poutine.util.site_is_subsample(node)
            ]
        )

        with pyro.poutine.trace() as model_tr:
            with block_guide_sample_sites:
                with pyro.poutine.replay(trace=guide_tr.trace), particles_plate:
                    self.model(*args, **kwargs)

        return {
            name: node["value"]
            for name, node in model_tr.trace.nodes.items()
            if node["type"] == "sample"
            and not pyro.poutine.util.site_is_subsample(node)
        }


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
    plate: pyro.poutine.indep_messenger.IndepMessenger

    def _pyro_sample(self, msg: dict) -> None:
        if msg["name"] not in self.data:
            return

        old_datum = torch.as_tensor(self.data[msg["name"]])
        event_dim = len(msg["fn"].event_shape)

        try:
            if not msg["infer"].get("_do_not_observe", None):
                new_datum: torch.Tensor = old_datum
                while len(new_datum.shape) - event_dim < self.plate.dim:
                    new_datum = new_datum[None]
                new_datum = new_datum.transpose(-len(old_datum.shape), self.plate.dim - event_dim)
                self.data[msg["name"]] = new_datum
                with self.plate:
                    return super()._pyro_sample(msg)
            else:
                return super()._pyro_sample(msg)
        finally:
            self.data[msg["name"]] = old_datum


def BatchedNMCLogPredictiveLikelihood(
    model: Callable[P, Any],
    guide: Callable[P, Any],
    *,
    max_plate_nesting: int,
    num_samples: int = 1,
    avg_particles: bool = False,
    data_plate_name: str = "__particles_data",
    mc_plate_name: str = "__particles_mc",
) -> Callable[Concatenate[Point[torch.Tensor], P], torch.Tensor]:

    def _fn(data: Point[torch.Tensor], *args: P.args, **kwargs: P.kwargs) -> torch.Tensor:

        plate_names: List[str] = []

        num_datapoints: int = next(iter(data.values())).shape[0]
        if num_datapoints > 1:
            data_plate = pyro.plate(data_plate_name, num_datapoints, dim=-max_plate_nesting - 2)
            data_cond = BatchedObservations(data=data, plate=data_plate)
            plate_names += [data_plate_name]
        else:
            data_cond = Observations(data=data)

        if num_samples > 1:
            particle_plate = pyro.plate(mc_plate_name, num_samples, dim=-max_plate_nesting - 1)
            plate_names += [mc_plate_name]
        else:
            particle_plate = contextlib.nullcontext()

        with particle_plate, data_cond:
            model_trace, guide_trace = pyro.infer.enum.get_importance_trace(
                "flat", max_plate_nesting, model, guide, args, kwargs
            )

        guide_trace.pack_tensors()
        model_trace.pack_tensors(guide_trace.plate_to_symbol)

        wds = "".join(guide_trace.plate_to_symbol[p] for p in plate_names)

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

        if mc_plate_name not in plate_names:
            log_weights = log_weights[None]

        if data_plate_name not in plate_names:
            log_weights = log_weights[..., None]

        if avg_particles:
            log_weights = torch.logsumexp(log_weights, dim=0) - math.log(num_samples)

        return log_weights

    return _fn
