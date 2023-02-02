import collections
import contextlib
import functools
import itertools
import numbers
from typing import (
    Callable,
    Container,
    Dict,
    FrozenSet,
    Generic,
    Iterable,
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

T = TypeVar("T")


class IndexSet(dict[str, Set[int]]):
    """
    A mapping from names to sets of indices.
    """

    def __init__(self, **mapping: Union[int, Iterable[int]]):
        super().__init__(
            **{
                k: {vs} if isinstance(vs, int) else set(vs)
                for k, vs in mapping.items()
                if vs
            }
        )

    def __hash__(self):
        return hash(self.as_relation(self))

    @classmethod
    def as_relation(cls, mapping: Dict[str, Set[int]]) -> FrozenSet[Tuple[str, int]]:
        return frozenset((k, v) for k, vs in mapping.items() for v in vs)

    @classmethod
    def from_relation(cls, relation: FrozenSet[Tuple[str, int]]) -> "IndexSet":
        return cls(
            **{
                k: {v for _, v in vs}
                for k, vs in itertools.groupby(sorted(relation), key=lambda x: x[0])
            }
        )

    @classmethod
    def meet(cls, *worlds: "IndexSet") -> "IndexSet":
        """
        Compute the intersection of multiple worlds.
        """
        return cls.from_relation(
            frozenset.intersection(*(map(cls.as_relation, worlds)))
        )

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
def gather(value, world: IndexSet, **kwargs):
    """
    Gather values from a single world in a multi-world object.
    """
    raise NotImplementedError


@functools.singledispatch
def scatter(value, world: IndexSet, *, result: Optional[T] = None, **kwargs):
    """
    Scatter values from multiple worlds into a single shared object.
    """
    raise NotImplementedError


def merge(partitioned_values: Dict[IndexSet, T], **kwargs) -> Optional[T]:
    """
    Merges a dictionary of dense values into a single value.

    :param dense_values: A dictionary mapping index sets to values.
    :return: A single value.
    """
    assert not functools.reduce(
        IndexSet.meet, partitioned_values.keys(), IndexSet()
    ), "keys must be disjoint"
    sparse_values = {k: gather(v, k, **kwargs) for k, v in partitioned_values.items()}
    result = None
    for indices, value in sparse_values.items():
        result = scatter(value, indices, result=result, **kwargs)
    return result
