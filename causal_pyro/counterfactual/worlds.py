import collections
import numbers
from typing import Dict, Hashable, List, Optional, TypeVar, Union

import pyro
import torch
from pyro.poutine.indep_messenger import CondIndepStackFrame, IndepMessenger

from .index_set import IndexSet, gather, indices_of, join, scatter

T = TypeVar("T")


@pyro.poutine.runtime.effectful(type="add_indices")
def add_indices(indexset: IndexSet) -> IndexSet:
    return indexset


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


def mask_as_indexset(
    mask: torch.Tensor,
    *,
    event_dim: int = 0,
    name_to_dim: Optional[Dict[Hashable, int]] = None,
) -> IndexSet:
    """
    Get a sparse index set from a dense mask.

    .. warning:: This is an expensive operation primarily useful for writing unit tests.
    """
    assert mask.dtype == torch.bool
    if name_to_dim is None:
        name_to_dim = {f.name: f.dim for f in get_index_plates().values()}
    raise NotImplementedError("TODO")


class _LazyPlateMessenger(IndepMessenger):
    @property
    def frame(self) -> CondIndepStackFrame:
        return CondIndepStackFrame(
            name=self.name, dim=self.dim, size=self.size, counter=0
        )

    def _process_message(self, msg):
        if msg["type"] not in (
            "sample",
            "param",
        ) or pyro.poutine.util.site_is_subsample(msg):
            return
        if self.frame.name in join(indices_of(msg["value"]), indices_of(msg["fn"])):
            super()._process_message(msg)


class IndexPlatesMessenger(pyro.poutine.messenger.Messenger):
    plates: Dict[Hashable, IndepMessenger]
    first_available_dim: int

    def __init__(self, first_available_dim: int):
        assert first_available_dim < 0
        self._orig_dim = first_available_dim
        self.first_available_dim = first_available_dim
        self.plates = collections.OrderedDict()
        super().__init__()

    def __enter__(self):
        self.first_available_dim = self._orig_dim
        self.plates = collections.OrderedDict()
        return super().__enter__()

    def __exit__(self, *args):
        for name in reversed(list(self.plates.keys())):
            self.plates.pop(name).__exit__(*args)
        return super().__exit__(*args)

    def _pyro_get_index_plates(self, msg):
        msg["value"] = {name: plate.frame for name, plate in self.plates.items()}
        msg["done"], msg["stop"] = True, True

    def _pyro_add_indices(self, msg):
        (indexset,) = msg["args"]
        for name, indices in indexset.items():
            if name not in self.plates:
                new_size = max(max(indices) + 1, len(indices))
                self.plates[name] = _LazyPlateMessenger(
                    name=name, dim=self.first_available_dim, size=new_size
                )
                self.plates[name].__enter__()
                self.first_available_dim -= 1
            else:
                assert (
                    0
                    <= min(indices)
                    <= len(indices) - 1
                    <= max(indices)
                    < self.plates[name].size
                ), f"cannot add {name}={indices} to {self.plates[name].size}"


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
def _indices_of_shape(value: torch.Size, **kwargs) -> IndexSet:
    return IndexSet(
        **{
            f.name: set(range(value[f.dim]))
            for f in get_index_plates().values()
            if -f.dim <= len(value) and value[f.dim] > 1
        }
    )


@indices_of.register
def _indices_of_tensor(value: torch.Tensor, *, event_dim: int = 0) -> IndexSet:
    batch_shape = value.shape[: len(value.shape) - event_dim]
    return indices_of(batch_shape)


@indices_of.register
def _indices_of_distribution(
    value: pyro.distributions.Distribution, **kwargs
) -> IndexSet:
    batch_shape = value.batch_shape
    return indices_of(batch_shape)


@indices_of.register
def _indices_of_maskeddist(
    value: pyro.distributions.MaskedDistribution, **kwargs
) -> IndexSet:
    return indices_of(value._mask)
