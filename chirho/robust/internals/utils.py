import contextlib
import functools
import math
from typing import Any, Callable, Dict, Generic, Mapping, Optional, Tuple, TypeVar

import pyro
import torch
from typing_extensions import Concatenate, ParamSpec

from chirho.indexed.handlers import add_indices
from chirho.indexed.ops import IndexSet, get_index_plates, indices_of
from chirho.observational.handlers.condition import Observations
from chirho.robust.ops import Point

P = ParamSpec("P")
Q = ParamSpec("Q")
S = TypeVar("S")
T = TypeVar("T")

ParamDict = Mapping[str, torch.Tensor]


@functools.singledispatch
def make_flatten_unflatten(
    v,
) -> Tuple[Callable[[T], torch.Tensor], Callable[[torch.Tensor], T]]:
    raise NotImplementedError


@make_flatten_unflatten.register(torch.Tensor)
def _make_flatten_unflatten_tensor(v: torch.Tensor):
    batch_size = v.shape[0]

    def flatten(v: torch.Tensor) -> torch.Tensor:
        r"""
        Flatten a tensor into a single vector.
        """
        return v.reshape((batch_size, -1))

    def unflatten(x: torch.Tensor) -> torch.Tensor:
        r"""
        Unflatten a vector into a tensor.
        """
        return x.reshape(v.shape)

    return flatten, unflatten


@make_flatten_unflatten.register(dict)
def _make_flatten_unflatten_dict(d: Dict[str, torch.Tensor]):
    batch_size = next(iter(d.values())).shape[0]

    def flatten(d: Dict[str, torch.Tensor]) -> torch.Tensor:
        r"""
        Flatten a dictionary of tensors into a single vector.
        """
        return torch.hstack([v.reshape((batch_size, -1)) for k, v in d.items()])

    def unflatten(x: torch.Tensor) -> Dict[str, torch.Tensor]:
        r"""
        Unflatten a vector into a dictionary of tensors.
        """
        return dict(
            zip(
                d.keys(),
                [
                    v_flat.reshape(v.shape)
                    for v, v_flat in zip(
                        d.values(),
                        torch.split(
                            x,
                            [int(v.numel() / batch_size) for k, v in d.items()],
                            dim=1,
                        ),
                    )
                ],
            )
        )

    return flatten, unflatten


def make_functional_call(
    mod: Callable[P, T]
) -> Tuple[ParamDict, Callable[Concatenate[ParamDict, P], T]]:
    assert isinstance(mod, torch.nn.Module)
    param_dict: ParamDict = dict(mod.named_parameters())

    @torch.func.functionalize
    def mod_func(params: ParamDict, *args: P.args, **kwargs: P.kwargs) -> T:
        with pyro.validation_enabled(False):
            return torch.func.functional_call(mod, params, args, dict(**kwargs))

    return param_dict, mod_func


@pyro.poutine.block()
@pyro.validation_enabled(False)
@torch.no_grad()
def guess_max_plate_nesting(
    model: Callable[P, Any], guide: Callable[P, Any], *args: P.args, **kwargs: P.kwargs
) -> int:
    elbo = pyro.infer.Trace_ELBO()
    elbo._guess_max_plate_nesting(model, guide, args, kwargs)
    return elbo.max_plate_nesting


@contextlib.contextmanager
def reset_rng_state(rng_state: T):
    try:
        prev_rng_state: T = pyro.util.get_rng_state()
        yield pyro.util.set_rng_state(rng_state)
    finally:
        pyro.util.set_rng_state(prev_rng_state)


@functools.singledispatch
def unbind_leftmost_dim(v, name: str, size: int = 1, **kwargs):
    raise NotImplementedError


@unbind_leftmost_dim.register
def _unbind_leftmost_dim_tensor(
    v: torch.Tensor, name: str, size: int = 1, *, event_dim: int = 0
) -> torch.Tensor:
    size = max(size, v.shape[0])
    v = v.expand((size,) + v.shape[1:])

    if name not in get_index_plates():
        add_indices(IndexSet(**{name: set(range(size))}))

    new_dim: int = get_index_plates()[name].dim
    orig_shape = v.shape
    while new_dim - event_dim < -len(v.shape):
        v = v[None]
    if v.shape[0] == 1 and orig_shape[0] != 1:
        v = torch.transpose(v, -len(orig_shape), new_dim - event_dim)
    return v


@unbind_leftmost_dim.register
def _unbind_leftmost_dim_distribution(
    v: pyro.distributions.Distribution, name: str, size: int = 1, **kwargs
) -> pyro.distributions.Distribution:
    size = max(size, v.batch_shape[0])
    if v.batch_shape[0] != 1:
        raise NotImplementedError("Cannot freely reshape distribution")

    if name not in get_index_plates():
        add_indices(IndexSet(**{name: set(range(size))}))

    new_dim: int = get_index_plates()[name].dim
    orig_shape = v.batch_shape

    new_shape = (size,) + (1,) * (-new_dim - len(orig_shape)) + orig_shape[1:]
    return v.expand(new_shape)


def get_importance_traces(
    model: Callable[P, Any],
    guide: Optional[Callable[P, Any]] = None,
) -> Callable[P, Tuple[pyro.poutine.Trace, pyro.poutine.Trace]]:
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
        if (
            self.num_particles > 1
            and msg["value"] is None
            and not pyro.poutine.util.site_is_factor(msg)
            and not pyro.poutine.util.site_is_subsample(msg)
            and self.name not in indices_of(msg["fn"])
        ):
            msg["fn"] = unbind_leftmost_dim(
                msg["fn"].expand((1,) + msg["fn"].batch_shape),
                self.name,
                size=self.num_particles,
            )
