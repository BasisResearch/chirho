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
    """
    Returns functions to flatten and unflatten an object. Used as a helper
    in :func:`chirho.robust.internals.linearize.conjugate_gradient_solve`

    :param v: some object
    :raises NotImplementedError:
    :return: flatten and unflatten functions
    :rtype: Tuple[Callable[[T], torch.Tensor], Callable[[torch.Tensor], T]]
    """
    raise NotImplementedError


@make_flatten_unflatten.register(torch.Tensor)
def _make_flatten_unflatten_tensor(v: torch.Tensor):
    """
    Returns functions to flatten and unflatten a `torch.Tensor`.
    """
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
    """
    Returns functions to flatten and unflatten a dictionary of `torch.Tensor`s.
    """
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
    """
    Converts a PyTorch module into a functional call for use with
    functions in :class:`torch.func`.

    :param mod: PyTorch module
    :type mod: Callable[P, T]
    :return: parameter dictionary and functional call
    :rtype: Tuple[ParamDict, Callable[Concatenate[ParamDict, P], T]]
    """
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
    """
    Guesses the maximum plate nesting level by running `pyro.infer.Trace_ELBO`

    :param model: Python callable containing Pyro primitives.
    :type model: Callable[P, Any]
    :param guide: Python callable containing Pyro primitives.
    :type guide: Callable[P, Any]
    :return: maximum plate nesting level
    :rtype: int
    """
    elbo = pyro.infer.Trace_ELBO()
    elbo._guess_max_plate_nesting(model, guide, args, kwargs)
    return elbo.max_plate_nesting


@contextlib.contextmanager
def reset_rng_state(rng_state: T):
    """
    Helper to temporarily reset the Pyro RNG state.
    """
    try:
        prev_rng_state: T = pyro.util.get_rng_state()
        yield pyro.util.set_rng_state(rng_state)
    finally:
        pyro.util.set_rng_state(prev_rng_state)
