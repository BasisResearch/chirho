from typing import (
    TYPE_CHECKING,
    Callable,
    FrozenSet,
    Generic,
    List,
    Optional,
    Protocol,
    Tuple,
    TypeVar,
    runtime_checkable,
)

if TYPE_CHECKING:
    from .handlers import DynamicInterruption, PointInterruption, Interruption

import functools

import pyro
import torch

S = TypeVar("S")
T = TypeVar("T")


@functools.singledispatch
def unsqueeze(x, axis: int):
    raise NotImplementedError(f"unsqueeze not implemented for type {type(x)}")


@unsqueeze.register
def _unsqueeze_torch(x: torch.Tensor, axis: int) -> torch.Tensor:
    return torch.unsqueeze(x, axis)


def _index_last_dim_with_mask(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    # Index into the last dimension of x with a boolean mask.
    # TODO AZ — There must be an easier way to do this?
    # NOTE AZ — this could be easily modified to support the last n dimensions, adapt if needed.

    if mask.dtype != torch.bool:
        raise ValueError(
            f"_index_last_dim_with_mask only supports boolean mask indexing, but got dtype {mask.dtype}."
        )

    # Require that the mask is 1d and aligns with the last dimension of x.
    if mask.ndim != 1 or mask.shape[0] != x.shape[-1]:
        raise ValueError(
            "_index_last_dim_with_mask only supports 1d boolean mask indexing, and must align with the last "
            f"dimension of x, but got mask shape {mask.shape} and x shape {x.shape}."
        )

    return torch.masked_select(
        x,
        # Get a shape that will broadcast to the shape of x. This will be [1, ..., len(mask)].
        mask.reshape((1,) * (x.ndim - 1) + mask.shape)
        # masked_select flattens tensors, so we need to reshape back to the original shape w/ the mask applied.
    ).reshape(x.shape[:-1] + (int(mask.sum()),))