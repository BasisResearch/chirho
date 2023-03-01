import numbers
from typing import Dict, Hashable, List, Optional, TypeVar, Union

import pyro
import torch
from pyro.poutine.indep_messenger import CondIndepStackFrame

from ..primitives import IndexSet, gather, indices_of, scatter, union

T = TypeVar("T")


@pyro.poutine.runtime.effectful(type="get_index_plates")
def get_index_plates() -> Dict[Hashable, CondIndepStackFrame]:
    raise NotImplementedError


def indexset_as_mask(
    indexset: IndexSet,
    *,
    event_dim: int = 0,
    name_to_dim: Optional[Dict[Hashable, int]] = None,
) -> torch.Tensor:
    """
    Get a dense mask tensor for indexing into a tensor from an indexset.
    """
    if name_to_dim is None:
        name_to_dim = {f.name: f.dim for f in get_index_plates().values()}
    batch_shape = [1] * -min(name_to_dim.values())
    inds: List[Union[slice, torch.Tensor]] = [slice(None)] * len(batch_shape)
    for name, values in indexset.items():
        inds[name_to_dim[name]] = torch.tensor(list(sorted(values)), dtype=torch.long)
        batch_shape[name_to_dim[name]] = max(len(values), max(values) + 1)
    mask = torch.zeros(tuple(batch_shape), dtype=torch.bool)
    mask[tuple(inds)] = True
    return mask[(...,) + (None,) * event_dim]


# Note that `gather` is defined using a `@functools.singledispatch` decorator,
# which in turn defines the `@gather.register` decorator used here
@gather.register
def _gather_number(
    value: numbers.Number,
    indexset: IndexSet,
    *,
    event_dim: Optional[int] = None,
    name_to_dim: Optional[Dict[Hashable, int]] = None,
) -> Union[numbers.Number, torch.Tensor]:
    assert event_dim is None or event_dim == 0
    return gather(
        torch.as_tensor(value), indexset, event_dim=event_dim, name_to_dim=name_to_dim
    )


@gather.register
def _gather_tensor(
    value: torch.Tensor,
    indexset: IndexSet,
    *,
    event_dim: Optional[int] = None,
    name_to_dim: Optional[Dict[Hashable, int]] = None,
) -> torch.Tensor:
    if event_dim is None:
        event_dim = 0

    if name_to_dim is None:
        name_to_dim = {f.name: f.dim for f in get_index_plates().values()}

    result = value
    for name, indices in indexset.items():
        dim = name_to_dim[name] - event_dim
        if len(result.shape) < -dim or result.shape[dim] == 1:
            continue
        result = result.index_select(
            name_to_dim[name] - event_dim,
            torch.tensor(list(sorted(indices)), device=value.device, dtype=torch.long),
        )
    return result


@scatter.register
def _scatter_number(
    value: numbers.Number,
    indexset: IndexSet,
    *,
    result: Optional[torch.Tensor] = None,
    event_dim: Optional[int] = None,
    name_to_dim: Optional[Dict[Hashable, int]] = None,
) -> Union[numbers.Number, torch.Tensor]:
    assert event_dim is None or event_dim == 0
    return scatter(
        torch.as_tensor(value),
        indexset,
        result=result,
        event_dim=event_dim,
        name_to_dim=name_to_dim,
    )


@scatter.register
def _scatter_tensor(
    value: torch.Tensor,
    indexset: IndexSet,
    *,
    result: Optional[torch.Tensor] = None,
    event_dim: Optional[int] = None,
    name_to_dim: Optional[Dict[Hashable, int]] = None,
) -> torch.Tensor:
    if event_dim is None:
        event_dim = 0

    if name_to_dim is None:
        name_to_dim = {f.name: f.dim for f in get_index_plates().values()}

    value = gather(value, indexset, event_dim=event_dim, name_to_dim=name_to_dim)
    indexset = union(
        indexset, indices_of(value, event_dim=event_dim, name_to_dim=name_to_dim)
    )

    if result is None:
        index_plates = get_index_plates()
        result_shape = list(
            torch.broadcast_shapes(
                value.shape,
                (1,) * max([event_dim - f.dim for f in index_plates.values()] + [0]),
            )
        )
        for name, indices in indexset.items():
            result_shape[name_to_dim[name] - event_dim] = index_plates[name].size
        result = value.new_zeros(result_shape)

    index: List[Union[slice, torch.Tensor]] = [slice(None)] * len(result.shape)
    for name, indices in indexset.items():
        if result.shape[name_to_dim[name] - event_dim] > 1:
            index[name_to_dim[name] - event_dim] = torch.tensor(
                list(sorted(indices)), device=value.device, dtype=torch.long
            )

    result[tuple(index)] = value
    return result


@indices_of.register
def _indices_of_number(value: numbers.Number, **kwargs) -> IndexSet:
    return IndexSet()


@indices_of.register
def _indices_of_bool(value: bool, **kwargs) -> IndexSet:
    return IndexSet()


@indices_of.register
def _indices_of_tuple(value: tuple, **kwargs) -> IndexSet:
    if all(isinstance(v, int) for v in value):
        return indices_of(torch.Size(value), **kwargs)
    return union(*(indices_of(v, **kwargs) for v in value))


@indices_of.register
def _indices_of_shape(value: torch.Size, **kwargs) -> IndexSet:
    name_to_dim = (
        kwargs["name_to_dim"]
        if "name_to_dim" in kwargs
        else {f.name: f.dim for f in get_index_plates().values()}
    )
    value = value[: len(value) - kwargs.get("event_dim", 0)]
    return IndexSet(
        **{
            name: set(range(value[dim]))
            for name, dim in name_to_dim.items()
            if -dim <= len(value) and value[dim] > 1
        }
    )


@indices_of.register
def _indices_of_tensor(value: torch.Tensor, **kwargs) -> IndexSet:
    return indices_of(value.shape, **kwargs)


@indices_of.register
def _indices_of_distribution(
    value: pyro.distributions.Distribution, **kwargs
) -> IndexSet:
    kwargs.pop("event_dim", None)
    return indices_of(value.batch_shape, event_dim=0, **kwargs)
