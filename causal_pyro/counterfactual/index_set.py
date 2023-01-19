import collections
import contextlib
import functools
import itertools
import numbers
from typing import Callable, Container, Dict, FrozenSet, Generic, Hashable, List, Mapping, \
    NamedTuple, Optional, Sequence, Set, Tuple, TypeVar, Union

import pyro
import torch

T = TypeVar("T")


class IndexSet(dict[Hashable, Set[int]]):
    """
    A mapping from names to sets of indices.
    """

    def __init__(self, **mapping: Union[int, Sequence[int]]):
        super().__init__(**{
            k: {vs} if isinstance(vs, int) else set(vs)
            for k, vs in mapping.items() if vs
        })

    def __hash__(self):
        return hash(self.as_relation(self))

    @classmethod
    def as_relation(cls, mapping: Dict[Hashable, Set[int]]) -> FrozenSet[Tuple[Hashable, int]]:
        return frozenset((k, v) for k, vs in mapping.items() for v in vs)

    @classmethod
    def from_relation(cls, relation: FrozenSet[Tuple[Hashable, int]]) -> "IndexSet":
        return cls(**{
            k: {v for _, v in vs}
            for k, vs in itertools.groupby(sorted(relation), key=lambda x: x[0])
        })

    @classmethod
    def as_mask(
        cls,
        mapping: "IndexSet",
        *,
        event_dim: int = 0,
        name_to_dim: Dict[Hashable, int] = {},
    ) -> torch.Tensor:
        """
        Get a mask for indexing into a world.
        """
        batch_shape = [1] * -min(name_to_dim.values())
        inds = [slice(None)] * len(batch_shape)
        for name, values in mapping.items():
            inds[name_to_dim[name]] = torch.tensor(list(sorted(values)), dtype=torch.long)
            batch_shape[name_to_dim[name]] = max(len(values), max(values) + 1)
        mask = torch.zeros(tuple(batch_shape), dtype=torch.bool)
        mask[tuple(inds)] = True
        return mask[(...,) + (None,) * event_dim]

    @classmethod
    def from_mask(
        cls,
        mask: torch.Tensor,
        *,
        event_dim: int = 0,
        name_to_dim: Dict[Hashable, int] = {}
    ) -> "IndexSet":
        """
        Get a world from a mask.
        """
        assert mask.dtype == torch.bool
        raise NotImplementedError("TODO")

    @classmethod
    def meet(cls, *worlds: "IndexSet") -> "IndexSet":
        """
        Compute the intersection of multiple worlds.
        """
        return cls.from_relation(frozenset.intersection(*(map(cls.as_relation, worlds))))

    @classmethod
    def join(cls, *worlds: "IndexSet") -> "IndexSet":
        """
        Compute the union of multiple worlds.
        """
        return cls.from_relation(frozenset.union(*map(cls.as_relation, worlds)))

    @classmethod
    def difference(cls, lhs: "IndexSet", rhs: "IndexSet") -> "IndexSet":
        """
        Compute the difference of two worlds.
        """
        return cls.from_relation(cls.as_relation(lhs) - cls.as_relation(rhs))


@functools.singledispatch
def indices_of(value, **kwargs) -> IndexSet:
    """
    Get the world of a value.
    """
    if callable(value) or value is None:
        return IndexSet()
    raise NotImplementedError


@functools.singledispatch
def gather(value: T, world: IndexSet, *, event_dim: Optional[int] = None) -> T:
    """
    Gather values from a single world in a multi-world object.
    """
    raise NotImplementedError


@functools.singledispatch
def scatter(value: T, world: IndexSet, *, result: Optional[T] = None, event_dim: Optional[int] = None) -> T:
    """
    Scatter values from multiple worlds into a single shared object.
    """
    raise NotImplementedError
