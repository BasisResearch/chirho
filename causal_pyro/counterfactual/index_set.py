"""
Index sets
==========

Index sets are used to represent sets of indices of elements in a list or array.
They are used to represent sets of indices of variables in a model, and sets of
indices of observations in a dataset.

Index sets are represented as a mapping from strings to sets of integers.
The strings are labels for the sets of indices, and the integers are the indices
of the elements in the list or array.

For example, the index set
```
{"x": {0, 1}, "y": {2, 3}}
```
represents the sets of indices of the variables "x" and "y" in a model.
"""
import functools
from typing import Dict, Iterable, Optional, Set, TypeVar, Union

T = TypeVar("T")


class IndexSet(dict[str, Set[int]]):
    """
    Data structure used to store labelled sets of integers, where the integers
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
        return hash(frozenset((k, v) for k, vs in self.items() for v in vs))


def unifiable(*indexsets: IndexSet) -> bool:
    """
    Check if indexsets can be unified.

    Two indexsets can be unified if they have the same values for all shared keys.
    """
    if len(indexsets) < 2:
        return True
    if len(indexsets) == 2:
        lhs, rhs = indexsets
        return all(lhs[v] == rhs[v] for v in set(lhs.keys()) & set(rhs.keys()))
    return all(unifiable(a, b) for a, b in zip(indexsets[:-1], indexsets[1:]))


def meet(*indexsets: IndexSet) -> IndexSet:
    """
    Compute the intersection of multiple indexsets.

    Example::

        meet(IndexSet(a={0, 1}), IndexSet(a={1, 2})) == IndexSet(a={1})

    .. note::

        ``meet``, ``join``, and ``difference`` satisfy several algebraic equations.

        ``meet`` is associative and commutative::

            meet(a, meet(b, c)) == meet(meet(a, b), c)
            meet(a, b) == meet(b, a)

        ``meet`` is idempotent::

            meet(a, a) == a
            meet(a, meet(a, b)) == meet(a, b)

        ``meet`` distributes over ``join``::

            meet(a, join(b, c)) == join(meet(a, b), meet(a, c))
    """
    return IndexSet(**{
        k: set.intersection(*[vs[k] for vs in indexsets if k in vs])
        for k in set.intersection(*map(set, indexsets))
    })


def join(*indexsets: IndexSet) -> IndexSet:
    """
    Compute the union of multiple indexsets.

    Example::

        join(IndexSet(a={0, 1}), IndexSet(a={1, 2})) == IndexSet(a={0, 1, 2})

    .. note::

        ``meet``, ``join``, and ``difference`` satisfy several algebraic equations.

        ``join`` is associative and commutative::

            join(a, join(b, c)) == join(join(a, b), c)
            join(a, b) == join(b, a)

        ``join`` is idempotent::

            join(a, a) == a
            join(a, join(a, b)) == join(a, b)

        ``join`` distributes over ``meet``::

            meet(a, join(b, c)) == join(meet(a, b), meet(a, c))
    """
    return IndexSet(**{
        k: set.union(*[vs[k] for vs in indexsets if k in vs])
        for k in set.union(*map(set, indexsets))
    })


def difference(lhs: IndexSet, rhs: IndexSet) -> IndexSet:
    """
    Compute the difference of two indexsets.

    Example::

        difference(IndexSet(a={0, 1}), IndexSet(a={1, 2})) == IndexSet(a={0})
    """
    return IndexSet(**{k: lhs[k] - rhs.get(k, set()) for k in lhs})


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
