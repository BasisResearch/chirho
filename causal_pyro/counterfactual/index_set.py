"""
Module for working with index sets.
"""
import functools
import itertools
from typing import (
    Dict,
    FrozenSet,
    Iterable,
    Optional,
    Set,
    Tuple,
    TypeVar,
    Union,
)

T = TypeVar("T")


class IndexSet(dict[str, Set[int]]):
    """
    This class is used to store sets of integers, where the integers
    represent the indices of the elements present in a list or array.
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
        return hash(indexset_as_relation(self))


def indexset_as_relation(mapping: IndexSet) -> FrozenSet[Tuple[str, int]]:
    """
    Convert an :class:``IndexSet`` to a relation.

    The indexset is a mapping from strings to sets of integers.
    This function converts the indexset to a relation, which is
    a :class:``set`` of pairs of strings and integers. Algebraic operations
    on indexsets are easy to implement in terms of Python set operations.
    """
    return frozenset((k, v) for k, vs in mapping.items() for v in vs)


def relation_as_indexset(relation: FrozenSet[Tuple[str, int]]) -> IndexSet:
    """
    Converts a relation into an IndexSet.

    The input relation is a set of tuples,
    where the first element of the tuple is a string and the second element is
    an integer. The output IndexSet is a dict where the keys are strings and the
    values are sets of integers.

    The keys of the output IndexSet are the same
    as the strings in the tuples in the input relation, and the values of the
    output IndexSet are the set of integers in the tuples in the input relation
    that are associated with the key.
    """
    return IndexSet(
        **{
            k: {v for _, v in vs}
            for k, vs in itertools.groupby(sorted(relation), key=lambda x: x[0])
        }
    )


def meet(*indexsets: IndexSet) -> IndexSet:
    """
    Compute the intersection of multiple indexsets.

    For example, the meet of the indexsets ``{"a": {0, 1}}`` and ``{"a": {1, 2}}``
    is ``{"a": {1}}``.

    Note::

        meet is associative and commutative:

        .. code-block:: python

            meet(a, meet(b, c)) == meet(meet(a, b), c)
            meet(a, b) == meet(b, a)

    Note::

        meet is idempotent:

        .. code-block:: python

            meet(a, a) == a
            meet(a, meet(a, b)) == meet(a, b)
    """
    return relation_as_indexset(
        frozenset.intersection(*(map(indexset_as_relation, indexsets)))
    )


def join(*indexsets: IndexSet) -> IndexSet:
    """
    Compute the union of multiple indexsets.

    For example, the join of the indexsets ``{"a": {0, 1}}`` and ``{"b": {0, 1}}``
    is ``{"a": {0, 1}, "b": {0, 1}}``.

    Note::

        join is associative and commutative:

        .. code-block:: python

            join(a, join(b, c)) == join(join(a, b), c)
            join(a, b) == join(b, a)
    """
    return relation_as_indexset(frozenset.union(*map(indexset_as_relation, indexsets)))


def difference(lhs: IndexSet, rhs: IndexSet) -> IndexSet:
    """
    Compute the difference of two indexsets.

    For example, the difference of the indexsets ``{"a": {0, 1}}`` and ``{"a": {1, 2}}``
    is ``{"a": {0}}``.
    """
    return relation_as_indexset(indexset_as_relation(lhs) - indexset_as_relation(rhs))


@functools.singledispatch
def indices_of(value, **kwargs) -> IndexSet:
    """
    Get the indexset of a value.

    Can be extended to new value types by registering an implementation
    for the type using :func:``functools.singledispatch``.
    """
    if callable(value) or value is None:
        return IndexSet()
    raise NotImplementedError


@functools.singledispatch
def gather(value, indexset: IndexSet, **kwargs):
    """
    Gather values from a single indexset in a multi-world object.

    Can be extended to new value types by registering an implementation
    for the type using :func:``functools.singledispatch``.
    """
    raise NotImplementedError


@functools.singledispatch
def scatter(value, indexset: IndexSet, *, result: Optional[T] = None, **kwargs):
    """
    Scatter values from multiple indexsets into a single shared object.

    This function takes a value and an indexset, and scatters the value
    into the indexset, returning a new object.

    Can be extended to new value types by registering an implementation
    for the type using :func:``functools.singledispatch``.

    Parameters
    ----------
    value : Any
        The value to scatter.
    indexset : IndexSet
        The indexset to scatter the value into.
    result : Optional[T], optional
        The result to store the scattered value in.
    **kwargs
        Additional keyword arguments that are used by the specific
        implementation.

    Returns
    -------
    T
        The value, scattered into the indexset.
    """
    raise NotImplementedError


def merge(partitioned_values: Dict[IndexSet, T], **kwargs) -> Optional[T]:
    """
    Merges a dictionary of disjoint masked values into a single value
    using repeated calls to :func:``scatter``.

    :param dense_values: A dictionary mapping index sets to values.
    :return: A single value.
    """
    assert not functools.reduce(
        meet, partitioned_values.keys(), IndexSet()
    ), "keys must be disjoint"
    sparse_values = {k: gather(v, k, **kwargs) for k, v in partitioned_values.items()}
    result = None
    for indices, value in sparse_values.items():
        result = scatter(value, indices, result=result, **kwargs)
    return result
