import contextlib
import functools
from typing import Any, Callable, Dict, Mapping, Tuple, TypeVar

import pyro
import torch
from typing_extensions import Concatenate, ParamSpec

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
    def flatten(v: torch.Tensor) -> torch.Tensor:
        r"""
        Flatten a tensor into a single vector.
        """
        return v.flatten()

    def unflatten(x: torch.Tensor) -> torch.Tensor:
        r"""
        Unflatten a vector into a tensor.
        """
        return x.reshape(v.shape)

    return flatten, unflatten


@make_flatten_unflatten.register(dict)
def _make_flatten_unflatten_dict(d: Dict[str, torch.Tensor]):
    def flatten(d: Dict[str, torch.Tensor]) -> torch.Tensor:
        r"""
        Flatten a dictionary of tensors into a single vector.
        """
        return torch.cat([v.flatten() for k, v in d.items()])

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
                        d.values(), torch.split(x, [v.numel() for k, v in d.items()])
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
