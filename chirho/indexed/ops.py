import functools
import operator
from typing import Dict, Hashable, Iterable, List, Optional, Set, Tuple, TypeVar, Union

import pyro
import torch

T = TypeVar("T")


class IndexSet(Dict[str, Set[int]]):
    """
    :class:`IndexSet` s represent the support of an indexed value, primarily
    those created using :func:`intervene` and :class:`MultiWorldCounterfactual`
    for which free variables correspond to single interventions and indices
    to worlds where that intervention either did or did not happen.

    :class:`IndexSet` can be understood conceptually as generalizing :class:`torch.Size`
    from multidimensional arrays to arbitrary values, from positional to named dimensions,
    and from bounded integer interval supports to finite sets of positive integers.

    :class:`IndexSet` s are implemented as :class:`dict` s with
    :class:`str` s as keys corresponding to names of free index variables
    and :class:`set` s of positive :class:`int` s as values corresponding
    to the values of the index variables where the indexed value is defined.

    For example, the following :class:`IndexSet` represents
    the sets of indices of the free variables ``x`` and ``y``
    for which a value is defined::

        >>> IndexSet(x={0, 1}, y={2, 3}})
        {"x": {0, 1}, "y": {2, 3}}

    :class:`IndexSet` 's constructor will automatically drop empty entries
    and attempt to convert input values to :class:`set` s::

        >>> IndexSet(x=[0, 0, 1], y=set(), z=2)
        {"x": {0, 1}, "z": {2}}

    :class:`IndexSet` s are also hashable and can be used as keys in :class:`dict` s::

        >>> indexset = IndexSet(x={0, 1}, y={2, 3}})
        >>> indexset in {indexset: 1}
        True
    """

    def __init__(self, **mapping: Union[int, Iterable[int]]):
        super().__init__(
            **{
                k: {vs} if isinstance(vs, int) else set(vs)
                for k, vs in mapping.items()
                if vs
            }
        )

    def __repr__(self):
        return f"{type(self).__name__}({super().__repr__()})"

    def __hash__(self):
        return hash(frozenset((k, frozenset(vs)) for k, vs in self.items()))


def union(*indexsets: IndexSet) -> IndexSet:
    """
    Compute the union of multiple :class:`IndexSet` s
    as the union of their keys and of value sets at shared keys.

    If :class:`IndexSet` may be viewed as a generalization of :class:`torch.Size`,
    then :func:`union` is a generalization of :func:`torch.broadcast_shapes`
    for the more abstract :class:`IndexSet` data structure.

    Example::

        >>> union(IndexSet(a={0, 1}, b={1}), IndexSet(a={1, 2}))
        {"a": {0, 1, 2}, "b": {1}}

    .. note::

        :func:`union` satisfies several algebraic equations for arbitrary inputs.
        In particular, it is associative, commutative, idempotent and absorbing::

            union(a, union(b, c)) == union(union(a, b), c)
            union(a, b) == union(b, a)
            union(a, a) == a
            union(a, union(a, b)) == union(a, b)
    """
    return IndexSet(
        **{
            k: set.union(*[vs[k] for vs in indexsets if k in vs])
            for k in set.union(*(set(vs) for vs in indexsets))
        }
    )


@functools.singledispatch
def indices_of(value, **kwargs) -> IndexSet:
    """
    Get a :class:`IndexSet` of indices on which an indexed value is supported.
    :func:`indices_of` is useful in conjunction with :class:`MultiWorldCounterfactual`
    for identifying the worlds where an intervention happened upstream of a value.

    For example, in a model with an outcome variable ``Y`` and a treatment variable
    ``T`` that has been intervened on, ``T`` and ``Y`` are both indexed by ``"T"``::

        >>> with MultiWorldCounterfactual():
        ...     X = pyro.sample("X", get_X_dist())
        ...     T = pyro.sample("T", get_T_dist(X))
        ...     T = intervene(T, t, name="T_ax")  # adds an index variable "T_ax"
        ...     Y = pyro.sample("Y", get_Y_dist(X, T))
        ...     assert indices_of(X) == IndexSet()
        ...     assert indices_of(T) == IndexSet(T_ax={0, 1})
        ...     assert indices_of(Y) == IndexSet(T_ax={0, 1})

    Just as multidimensional arrays can be expanded to shapes with new dimensions
    over which they are constant, :func:`indices_of` is defined extensionally,
    meaning that values are treated as constant functions of free variables
    not in their support.

    .. note::

        :func:`indices_of` can be extended to new value types by registering
        an implementation for the type using :func:`functools.singledispatch` .

    .. note::

        Fully general versions of :func:`indices_of` , :func:`gather`
        and :func:`scatter` would require a dependent broadcasting semantics
        for indexed values, as is the case in sparse or masked array libraries
        like ``torch.sparse`` or relational databases.

        However, this is beyond the scope of this library as it currently exists.
        Instead, :func:`gather` currently binds free variables in its input indices
        when their indices there are a strict subset of the corresponding indices
        in ``value`` , so that they no longer appear as free in the result.

        For example, in the above snippet, applying :func:`gather` to to select only
        the values of ``Y`` from worlds where no intervention on ``T`` happened
        would result in a value that no longer contains free variable ``"T"``::

            >>> indices_of(Y) == IndexSet(T_ax={0, 1})
            True
            >>> Y0 = gather(Y, IndexSet(T_ax={0}))
            >>> indices_of(Y0) == IndexSet() != IndexSet(T_ax={0})
            True

        The practical implications of this imprecision are limited
        since we rarely need to :func:`gather` along a variable twice.

    :param value: A value.
    :param kwargs: Additional keyword arguments used by specific implementations.
    :return: A :class:`IndexSet` containing the indices on which the value is supported.
    """
    raise NotImplementedError


@functools.singledispatch
def gather(value, indexset: IndexSet, **kwargs):
    """
    Selects entries from an indexed value at the indices in a :class:`IndexSet` .
    :func:`gather` is useful in conjunction with :class:`MultiWorldCounterfactual`
    for selecting components of a value corresponding to specific counterfactual worlds.

    For example, in a model with an outcome variable ``Y`` and a treatment variable
    ``T`` that has been intervened on, we can use :func:`gather` to define quantities
    like treatment effects that require comparison of different potential outcomes::

        >>> with MultiWorldCounterfactual():
        ...     X = pyro.sample("X", get_X_dist())
        ...     T = pyro.sample("T", get_T_dist(X))
        ...     T = intervene(T, t, name="T_ax")  # adds an index variable "T_ax"
        ...     Y = pyro.sample("Y", get_Y_dist(X, T))
        ...     Y_factual = gather(Y, IndexSet(T_ax=0))         # no intervention
        ...     Y_counterfactual = gather(Y, IndexSet(T_ax=1))  # intervention
        ...     treatment_effect = Y_counterfactual - Y_factual

    Like :func:`torch.gather` and substitution in term rewriting,
    :func:`gather` is defined extensionally, meaning that values
    are treated as constant functions of variables not in their support.

    :func:`gather` will accordingly ignore variables in ``indexset``
    that are not in the support of ``value`` computed by :func:`indices_of` .

    .. note::

        :func:`gather` can be extended to new value types by registering
        an implementation for the type using :func:`functools.singledispatch` .

    .. note::

        Fully general versions of :func:`indices_of` , :func:`gather`
        and :func:`scatter` would require a dependent broadcasting semantics
        for indexed values, as is the case in sparse or masked array libraries
        like ``scipy.sparse`` or ``xarray`` or in relational databases.

        However, this is beyond the scope of this library as it currently exists.
        Instead, :func:`gather` currently binds free variables in ``indexset``
        when their indices there are a strict subset of the corresponding indices
        in ``value`` , so that they no longer appear as free in the result.

        For example, in the above snippet, applying :func:`gather` to to select only
        the values of ``Y`` from worlds where no intervention on ``T`` happened
        would result in a value that no longer contains free variable ``"T"``::

            >>> indices_of(Y) == IndexSet(T_ax={0, 1})
            True
            >>> Y0 = gather(Y, IndexSet(T_ax={0}))
            >>> indices_of(Y0) == IndexSet() != IndexSet(T_ax={0})
            True

        The practical implications of this imprecision are limited
        since we rarely need to :func:`gather` along a variable twice.

    :param value: The value to gather.
    :param IndexSet indexset: The :class:`IndexSet` of entries to select from ``value``.
    :param kwargs: Additional keyword arguments used by specific implementations.
    :return: A new value containing entries of ``value`` from ``indexset``.
    """
    raise NotImplementedError


@functools.singledispatch
def scatter(
    value, indexset: Optional[IndexSet] = None, *, result: Optional[T] = None, **kwargs
):
    """
    Assigns entries from an indexed value to entries in a larger indexed value.
    :func:`scatter` is primarily used internally in :class:`MultiWorldCounterfactual`
    for concisely and extensibly defining the semantics of counterfactuals.

    It also satisfies some equations with :func:`gather` and :func:`indices_of`
    that are useful for testing and debugging.

    Like :func:`torch.scatter` and assignment in term rewriting,
    :func:`scatter` is defined extensionally, meaning that values
    are treated as constant functions of variables not in their support.

    .. note::

        :func:`scatter` can be extended to new value types by registering
        an implementation for the type using :func:`functools.singledispatch` .

    :param value: The value to scatter.
    :param IndexSet indexset: The :class:`IndexSet` of entries of ``result`` to fill.
    :param Optional[T] result: The result to store the scattered value in.
    :param kwargs: Additional keyword arguments used by specific implementations.
    :return: The ``result``, with ``value`` scattered into the indices in ``indexset``.
    """
    raise NotImplementedError


@pyro.poutine.runtime.effectful(type="scatter_n")
def scatter_n(values: Dict[IndexSet, T], *, result: Optional[T] = None, **kwargs):
    """
    Scatters a dictionary of disjoint masked values into a single value
    using repeated calls to :func:``scatter``.

    :param partitioned_values: A dictionary mapping index sets to values.
    :return: A single value.
    """
    assert len(values) > 0
    assert all(isinstance(k, IndexSet) for k in values)
    for indices, value in values.items():
        result = scatter(value, indices, result=result, **kwargs)
    return result


@functools.singledispatch
def cond(fst, snd, case: Optional[T] = None, **kwargs):
    """
    Selection operation that is the sum-type analogue of :func:`scatter`
    in the sense that where :func:`scatter` propagates both of its arguments,
    :func:`cond` propagates only one, depending on the value of a boolean ``case`` .

    For a given ``fst`` , ``snd`` , and ``case`` , :func:`cond` returns
    ``snd`` if the ``case`` is true, and ``fst`` otherwise,
    analogous to a Python conditional expression ``snd if case else fst`` .
    Unlike a Python conditional expression, however, the case may be a tensor,
    and both branches are evaluated, as with :func:`torch.where` ::

        >>> fst, snd = torch.randn(2, 3), torch.randn(2, 3)
        >>> case = (fst < snd).all(-1)
        >>> x = cond(fst, snd, case, event_dim=1)
        >>> assert (x == torch.where(case[..., None], snd, fst)).all()

    .. note::

        :func:`cond` can be extended to new value types by registering
        an implementation for the type using :func:`functools.singledispatch` .

    :param fst: The value to return if ``case`` is ``False`` .
    :param snd: The value to return if ``case`` is ``True`` .
    :param case: A boolean value or tensor. If a tensor, should have event shape ``()`` .
    :param kwargs: Additional keyword arguments used by specific implementations.
    """
    raise NotImplementedError(f"cond not implemented for {type(fst)}")


@pyro.poutine.runtime.effectful(type="cond_n")
def cond_n(values: Dict[IndexSet, T], case: Union[bool, torch.Tensor], **kwargs):
    assert len(values) > 0
    assert all(isinstance(k, IndexSet) for k in values.keys())
    result: Optional[T] = None
    for indices, value in values.items():
        tst = torch.as_tensor(
            functools.reduce(
                operator.or_, [case == index for index in next(iter(indices.values()))]
            ),
            dtype=torch.bool,
        )
        result = cond(result if result is not None else value, value, tst, **kwargs)
    return result


@pyro.poutine.runtime.effectful(type="get_index_plates")
def get_index_plates() -> (
    Dict[Hashable, pyro.poutine.indep_messenger.CondIndepStackFrame]
):
    return {}


def indexset_as_mask(
    indexset: IndexSet,
    *,
    event_dim: int = 0,
    name_to_dim_size: Optional[Dict[Hashable, Tuple[int, int]]] = None,
    device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    """
    Get a dense mask tensor for indexing into a tensor from an indexset.
    """
    if name_to_dim_size is None:
        name_to_dim_size = {
            name: (f.dim, f.size) for name, f in get_index_plates().items()
        }
    batch_shape = [1] * -min([dim for dim, _ in name_to_dim_size.values()], default=0)
    inds: List[Union[slice, torch.Tensor]] = [slice(None)] * len(batch_shape)
    for name, values in indexset.items():
        dim, size = name_to_dim_size[name]
        inds[dim] = torch.tensor(list(sorted(values)), dtype=torch.long)
        batch_shape[dim] = size
    mask = torch.zeros(tuple(batch_shape), dtype=torch.bool, device=device)
    mask[tuple(inds)] = True
    return mask[(...,) + (None,) * event_dim]
