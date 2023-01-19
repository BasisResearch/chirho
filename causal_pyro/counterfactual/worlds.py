import collections
import functools
import numbers
from typing import (
    Callable,
    Container,
    Dict,
    FrozenSet,
    Generic,
    Hashable,
    List,
    Mapping,
    NamedTuple,
    Optional,
    Sequence,
    Set,
    Tuple,
    TypeVar,
    Union,
)

import pyro
import torch
from pyro.infer.reparam.reparam import Reparam
from pyro.infer.reparam.strategies import Strategy
from pyro.poutine.indep_messenger import CondIndepStackFrame, IndepMessenger

from .index_set import IndexSet, gather, indices_of, scatter

T, I = TypeVar("T"), TypeVar("I")


def complement(world: IndexSet) -> IndexSet:
    """
    Compute the complement of a world.
    """
    diff = IndexSet.difference(get_full_index(), world)
    # drop any axes that are still full, i.e. were not in world
    return IndexSet(
        **{name: indices for name, indices in diff.items() if name in world}
    )


@pyro.poutine.runtime.effectful(type="get_full_index")
def get_full_index() -> IndexSet:
    """
    Compute the full world.
    """
    raise NotImplementedError


@pyro.poutine.runtime.effectful(type="get_index_plates")
def get_index_plates() -> Dict[Hashable, CondIndepStackFrame]:
    raise NotImplementedError


@pyro.poutine.runtime.effectful(type="add_indices")
def add_indices(world: IndexSet) -> IndexSet:
    return world


def indexset_as_mask(world: IndexSet, *, event_dim: int = 0) -> torch.Tensor:
    """
    Get a mask for indexing into a world.
    """
    name_to_dim = {f.name: f.dim for f in get_index_plates().values()}
    batch_shape = [1] * -min(name_to_dim.values())
    inds = [slice(None)] * len(batch_shape)
    for name, values in world.items():
        inds[name_to_dim[name]] = torch.tensor(list(sorted(values)), dtype=torch.long)
        batch_shape[name_to_dim[name]] = max(len(values), max(values) + 1)
    mask = torch.zeros(tuple(batch_shape), dtype=torch.bool)
    mask[tuple(inds)] = True
    return mask[(...,) + (None,) * event_dim]


def mask_as_indexset(mask: torch.Tensor, *, event_dim: int = 0) -> IndexSet:
    """
    Get a sparse index set from a dense mask.
    """
    assert mask.dtype == torch.bool
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
        if self.frame.name in IndexSet.join(
            indices_of(msg["value"]), indices_of(msg["fn"])
        ):
            super()._process_message(msg)


class IndexPlatesMessenger(pyro.poutine.messenger.Messenger):
    plates: Dict[Hashable, IndepMessenger]
    first_available_dim: int

    def __init__(self, first_available_dim: int):
        assert first_available_dim < 0
        self._orig_dim = first_available_dim
        self.first_available_dim = first_available_dim
        self.plates = {}
        super().__init__()

    def __enter__(self):
        self.first_available_dim = self._orig_dim
        self.plates = collections.OrderedDict()
        return super().__enter__()

    def __exit__(self, *args):
        for name in reversed(list(self.plates.keys())):
            self.plates.pop(name).__exit__(*args)
        return super().__exit__(*args)

    def _pyro_get_full_index(self, msg):
        msg["value"] = IndexSet(
            **{name: set(range(plate.size)) for name, plate in self.plates.items()}
        )
        msg["stop"], msg["done"] = True, True

    def _pyro_get_index_plates(self, msg):
        msg["value"] = {name: plate.frame for name, plate in self.plates.items()}
        msg["done"], msg["stop"] = True, True

    def _pyro_add_indices(self, msg):
        (world,) = msg["args"]
        for name, indices in world.items():
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
                    <= len(indices)
                    <= max(indices)
                    <= self.plates[name].size
                ), f"cannot add {name}={indices} to {self.plates[name].size}"


@gather.register
def _gather_tensor(
    value: torch.Tensor, world: IndexSet, *, event_dim: Optional[int] = None
) -> torch.Tensor:
    if event_dim is None:
        event_dim = 0
    mask = indexset_as_mask(world, event_dim=event_dim).to(device=value.device)
    _, mask = torch.broadcast_tensors(value, mask)
    return value[mask]


@scatter.register
def _scatter_tensor(
    value: torch.Tensor,
    world: IndexSet,
    *,
    result: Optional[torch.Tensor] = None,
    event_dim: Optional[int] = None,
) -> torch.Tensor:
    if event_dim is None:
        event_dim = 0

    mask = indexset_as_mask(world, event_dim=event_dim).to(device=value.device)
    if result is None:
        result = value.new_zeros(
            mask.shape + value.shape[len(value.shape) - event_dim :]
        )
    _, mask = torch.broadcast_tensors(result, mask)
    result[mask] = value
    return result


@indices_of.register
def _indices_of_number(value: numbers.Number, **kwargs) -> IndexSet:
    return IndexSet()


@indices_of.register
def _indices_of_bool(value: bool, **kwargs) -> IndexSet:
    return IndexSet()


@indices_of.register
def _indices_of_tuple(value: tuple, **kwargs) -> IndexSet:
    if not value:
        return IndexSet()
    if all(isinstance(v, int) for v in value):
        return indices_of(torch.Size(value))
    return IndexSet.join(*map(indices_of, value))


@indices_of.register
def _indices_of_shape(value: torch.Size, **kwargs) -> IndexSet:
    return IndexSet(
        **{
            f.name: set(range(value[f.dim]))
            for f in get_index_plates().values()
            if value[f.dim] > 1
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
