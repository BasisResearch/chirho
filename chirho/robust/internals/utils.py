import contextlib
import functools
import math
from typing import Any, Callable, Dict, Mapping, Optional, Tuple, TypeVar

import pyro
import torch
from typing_extensions import Concatenate, ParamSpec

from chirho.indexed.handlers import add_indices
from chirho.indexed.ops import IndexSet, get_index_plates, indices_of

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


@functools.singledispatch
def unbind_leftmost_dim(v, name: str, size: int = 1, **kwargs):
    """
    Helper function to move the leftmost dimension of a ``torch.Tensor``
    or ``pyro.distributions.Distribution`` or other batched value
    into a fresh named dimension using the machinery in ``chirho.indexed`` ,
    allocating a new dimension with the given name if necessary
    via an enclosing :class:`~chirho.indexed.handlers.IndexPlatesMessenger` .

    .. warning:: Must be used in conjunction with :class:`~chirho.indexed.handlers.IndexPlatesMessenger` .

    :param v: Batched value.
    :param name: Name of the fresh dimension.
    :param size: Size of the fresh dimension. If 1, the size is inferred from ``v`` .
    """
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


@functools.singledispatch
def bind_leftmost_dim(v, name: str, **kwargs):
    """
    Helper function to move a named dimension managed by ``chirho.indexed``
    into a new unnamed dimension to the left of all named dimensions in the value.

    .. warning:: Must be used in conjunction with :class:`~chirho.indexed.handlers.IndexPlatesMessenger` .
    """
    raise NotImplementedError


@bind_leftmost_dim.register
def _bind_leftmost_dim_tensor(
    v: torch.Tensor, name: str, *, event_dim: int = 0, **kwargs
) -> torch.Tensor:
    if name not in indices_of(v, event_dim=event_dim):
        return v
    return torch.transpose(
        v[None], -len(v.shape) - 1, get_index_plates()[name].dim - event_dim
    )


def get_importance_traces(
    model: Callable[P, Any],
    guide: Optional[Callable[P, Any]] = None,
) -> Callable[P, Tuple[pyro.poutine.Trace, pyro.poutine.Trace]]:
    """
    Thin functional wrapper around :func:`~pyro.infer.enum.get_importance_trace`
    that cleans up the original interface to avoid unnecessary arguments
    and efficiently supports using the prior in a model as a default guide.

    :param model: Model to run.
    :param guide: Guide to run. If ``None``, use the prior in ``model`` as a guide.
    :returns: A function that takes the same arguments as ``model`` and ``guide`` and returns
        a tuple of importance traces ``(model_trace, guide_trace)``.
    """

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


def site_is_delta(msg: dict) -> bool:
    d = msg["fn"]
    while hasattr(d, "base_dist"):
        d = d.base_dist
    return isinstance(d, pyro.distributions.Delta)
