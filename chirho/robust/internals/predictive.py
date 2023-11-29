import contextlib
import math
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


class DiceMultiFrameTensor(pyro.infer.util.MultiFrameTensor):

    def sum_to(
        self,
        target_frames: Container[pyro.poutine.indep_messenger.CondIndepStackFrame],
        event_dim: int = 0
    ) -> torch.Tensor:
        log_q = super().sum_to(target_frames)
        log_dice = log_q - log_q.detach()
        return log_dice.expand(log_dice.shape + (1,) * event_dim)


@torch.func.functionalize
@pyro.validation_enabled(False)
def _dice_importance_weights(
    model_trace: pyro.poutine.Trace,
    guide_trace: pyro.poutine.Trace,
    *,
    particle_plate_name: str,
) -> torch.Tensor:

    model_trace.compute_log_prob()
    guide_trace.compute_log_prob()
    plate_stacks = pyro.infer.util.get_plate_stacks(model_trace)
    plate_stacks.update(pyro.infer.util.get_plate_stacks(guide_trace))
    assert all(any(f.name == particle_plate_name for f in fs) for fs in plate_stacks.values())

    log_dice = DiceMultiFrameTensor()
    log_weights = torch.zeros_like(model_trace.log_prob_sum())

    for name, node in guide_trace.nodes.items():
        if node["type"] == "sample" and not pyro.poutine.util.site_is_subsample(node):
            if not node["is_observed"] and not node["fn"].has_rsample:
                log_dice.add((plate_stacks[name], node["log_prob"]))

            node_weight = log_dice.sum_to(plate_stacks[name]) * node["log_prob"]
            for f in plate_stacks[name]:
                if f.name != particle_plate_name:
                    node_weight = node_weight.sum(dim=f.dim, keepdim=True)

            log_weights = log_weights - node_weight.reshape(-1)

    for name, node in model_trace.nodes.items():
        if node["type"] == "sample" and not pyro.poutine.util.site_is_subsample(node):
            if name not in guide_trace.nodes and not node["is_observed"] and not node["fn"].has_rsample:
                log_dice.add((plate_stacks[name], node["log_prob"]))

            node_weight = log_dice.sum_to(plate_stacks[name]) * node["log_prob"]
            for f in plate_stacks[name]:
                if f.name != particle_plate_name:
                    node_weight = node_weight.sum(dim=f.dim, keepdim=True)

            log_weights = log_weights + node_weight.reshape(-1)

    return log_weights


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
        if self.max_plate_nesting is None and self.num_samples > 1:
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
    particle_plate_name: str = "__predictive_particles"

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
        self._predictive_model = PredictiveFunctional(
            self.model, self.guide, num_samples=1, max_plate_nesting=max_plate_nesting
        )

    def forward(
        self, data: Point[T], *args: P.args, **kwargs: P.kwargs
    ) -> torch.Tensor:
        if self.max_plate_nesting is None:
            self.max_plate_nesting = guess_max_plate_nesting(
                self.model, self.guide, *args, **kwargs
            )

        with pyro.plate(self.particle_plate_name, self.num_samples, dim=-self.max_plate_nesting - 1):
            with pyro.poutine.trace() as guide_tr:
                self.guide(*args, **kwargs)

            with pyro.poutine.trace() as model_tr:
                with pyro.poutine.replay(trace=guide_tr.trace):
                    with condition(data=data):
                        self._predictive_model(*args, **kwargs)

        log_weights = _dice_importance_weights(model_tr.trace, guide_tr.trace, particle_plate_name=self.particle_plate_name)
        return torch.logsumexp(log_weights, dim=0) - math.log(self.num_samples)
