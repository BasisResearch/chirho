import contextlib
import math
from typing import Container, Generic, Optional, ParamSpec, TypeVar

import pyro
import torch

from chirho.indexed.handlers import DependentMaskMessenger
from chirho.observational.handlers import condition
from chirho.robust.internals.utils import guess_max_plate_nesting
from chirho.robust.ops import Model, Point

pyro.settings.set(module_local_params=True)

P = ParamSpec("P")
Q = ParamSpec("Q")
S = TypeVar("S")
T = TypeVar("T")


class UnmaskNamedSites(DependentMaskMessenger):
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
    model: Model[P]
    guide: Model[P]
    num_samples: int
    max_plate_nesting: Optional[int]

    def __init__(
        self,
        model: Model[P],
        guide: Model[P],
        *,
        num_samples: int = 1,
        max_plate_nesting: Optional[int] = None,
    ):
        super().__init__()
        assert isinstance(model, torch.nn.Module)
        assert isinstance(guide, torch.nn.Module)
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
    model: Model[P]
    guide: Model[P]
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

        masked_guide = pyro.poutine.mask(mask=False)(self.guide)
        masked_model = UnmaskNamedSites(names=set(data.keys()))(
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
