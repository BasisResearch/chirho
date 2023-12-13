import contextlib
import math
import warnings
from typing import Any, Callable, Container, Generic, Optional, TypeVar

import pyro
import torch
from typing_extensions import ParamSpec

from chirho.indexed.handlers import DependentMaskMessenger
from chirho.observational.handlers import condition
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


class PointLogPredictiveLikelihood(NMCLogPredictiveLikelihood):
    def forward(
        self, data: Point[T], *args: P.args, **kwargs: P.kwargs
    ) -> torch.Tensor:
        if self.max_plate_nesting is None:
            self.max_plate_nesting = guess_max_plate_nesting(
                self.model, self.guide, *args, **kwargs
            )

        # Retrieve point estimate by sampling from the guide once
        with pyro.poutine.trace() as guide_tr:
            self.guide(*args, **kwargs)

        point_estimate = {k: v["value"] for k, v in guide_tr.trace.nodes.items()}
        model_at_point = condition(data=point_estimate)(self.model)

        # Add plate to batch over many Monte Carlo draws from model
        num_monte_carlo = data[next(iter(data))].shape[0]  # type: ignore

        def vectorize(fn):
            def _fn(*args, **kwargs):
                with pyro.plate(
                    "__monte_carlo_samples",
                    size=num_monte_carlo,
                    dim=-self.max_plate_nesting - 1,
                ):
                    return fn(*args, **kwargs)

            return _fn

        batched_model = condition(data=data)(vectorize(model_at_point))

        # Compute log likelihood at each monte carlo sample
        log_like_trace = pyro.poutine.trace(batched_model).get_trace(*args, **kwargs)
        log_like_trace.compute_log_prob(lambda name, site: name in data.keys())
        log_prob_at_datapoints = torch.zeros(num_monte_carlo)
        for site_name in data.keys():
            # Sum probabilities over all dimensions except first batch dimension
            dims_to_sum = list(
                range(1, log_like_trace.nodes[site_name]["log_prob"].dim())
            )
            log_prob_at_datapoints += log_like_trace.nodes[site_name]["log_prob"].sum(
                dim=dims_to_sum
            )
        return log_prob_at_datapoints
