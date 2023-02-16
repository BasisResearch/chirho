"""
Design notes: Interventions on values
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Pyro's design makes extensive use of `algebraic effect handlers <http://pyro.ai/examples/effect_handlers.html>`_,
a technology from programming language research for representing side effects compositionally.
As described in the `Pyro introductory tutorial <http://pyro.ai/examples/intro_long.html>`_,
sampling or observing a random variable is done using the ``pyro.sample`` primitive,
whose behavior is modified by effect handlers during posterior inference::

    @pyro.poutine.runtime.effectful
    def sample(name: str, dist: pyro.distributions.Distribution, obs: Optional[Tensor] = None) -> Tensor:
        return obs if obs is not None else dist.sample()

Pyro already has an effect handler `pyro.poutine.do` for intervening on `sample` statements,
but its implementation is too limited to be ergonomic for most causal inference problems of interest to practitioners::

   def model():
       x = sample("x", Normal(0, 1))
       y = sample("y", Normal(x, 1))
       return x, y

   x, y = model()
   assert x != 10  # with probability 1

   with pyro.poutine.do({"x": 10}):
       x, y = model()
   assert x == 10

This library is built around understanding interventions as side-effectful operations on values within a Pyro model::

   @effectful
   def intervene(obs: T, act: Intervention[T]) -> T:
       return act

The polymorphic definition of `intervene` above can be expanded as the generic type `Intervention` is made explicit::

   T = TypeVar("T", bound=[Number, Tensor, Callable])

   Intervention = Union[
       Optional[T],
       Callable[[T], T]
   ]

   @pyro.poutine.runtime.effectful(type="intervene")
   def intervene(obs: T, act: Intervention[T]) -> T:
       if act is None:
           return obs
       elif callable(act):
           return act(obs)
       else:
           return act

"""
import functools
from typing import Callable, Iterable, Optional, Set, TypeVar, Union

import pyro

T = TypeVar("T")

Intervention = Union[
    Optional[T],
    Callable[[T], T],
]


@pyro.poutine.runtime.effectful(type="intervene")
def intervene(
    obs: T, act: Intervention[T] = None, *, event_dim: Optional[int] = None
) -> T:
    """
    Intervene on a value in a probabilistic program.
    """
    if callable(act) and not isinstance(act, pyro.distributions.Distribution):
        return act(obs)
    elif act is None:
        return obs
    return act


class IndexSet(dict[str, Set[int]]):
    """
    :class:`IndexSet` s represent the support of an indexed value, primarily
    those created using :func:`intervene` and :class:`MultiWorldCounterfactual`
    for which free variables correspond to single interventions and indices
    to worlds where that intervention either did or did not happen.

    :class:`IndexSet` can be understood conceptually
    as generalizing :class:`torch.Size`
    from multidimensional arrays to arbitrary values,
    from positional to named dimensions,
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

    def __hash__(self):
        return hash(frozenset((k, frozenset(vs)) for k, vs in self.items()))


def join(*indexsets: IndexSet) -> IndexSet:
    """
    Compute the union of multiple :class:`IndexSet` s
    as the union of their keys and of value sets at shared keys.

    If :class:`IndexSet` may be viewed as a generalization of :class:`torch.Size`,
    then :func:`join` is a generalization of :func:`torch.broadcast_shapes`
    for the more abstract :class:`IndexSet` data structure.

    Example::

        >>> join(IndexSet(a={0, 1}, b={1}), IndexSet(a={1, 2}))
        {"a": {0, 1, 2}, "b": {1}}

    .. note::
        :func:`join` satisfies several algebraic equations for arbitrary inputs.
        In particular, it is associative, commutative, idempotent and absorbing::

            join(a, join(b, c)) == join(join(a, b), c)
            join(a, b) == join(b, a)
            join(a, a) == a
            join(a, join(a, b)) == join(a, b)
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
        ...     T = intervene(T, t, name="T")
        ...     Y = pyro.sample("Y", get_Y_dist(X, T))

        ...     assert indices_of(X) == IndexSet()
        ...     assert indices_of(T) == IndexSet(T={0, 1})
        ...     assert indices_of(Y) == IndexSet(T={0, 1})

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
        Instead, :func:`gather` currently binds free variables in ``indexset``
        when their indices there are a strict subset of the corresponding indices
        in ``value`` , so that they no longer appear as free in the result.

        For example, in the above snippet, applying :func:`gather` to to select only
        the values of ``Y`` from worlds where no intervention on ``T`` happened
        would result in a value that no longer contains free variable ``"T"``::

            >>> indices_of(Y) == IndexSet(T={0, 1})
            True
            >>> Y0 = gather(Y, IndexSet(T={0}))
            >>> indices_of(Y0) == IndexSet() != IndexSet(T={0})
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
        ...     T = intervene(T, t, name="T")
        ...     Y = pyro.sample("Y", get_Y_dist(X, T))

        ...     Y_factual = gather(Y, IndexSet(T=0))         # no intervention
        ...     Y_counterfactual = gather(Y, IndexSet(T=1))  # intervention
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

            >>> indices_of(Y) == IndexSet(T={0, 1})
            True
            >>> Y0 = gather(Y, IndexSet(T={0}))
            >>> indices_of(Y0) == IndexSet() != IndexSet(T={0})
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
def scatter(value, indexset: IndexSet, *, result: Optional[T] = None, **kwargs):
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


def merge(partitioned_values: dict[IndexSet, T], **kwargs) -> Optional[T]:
    """
    Merges a dictionary of disjoint masked values into a single value
    using repeated calls to :func:``scatter``.

    :param dense_values: A dictionary mapping index sets to values.
    :return: A single value.
    """
    result = None
    for indices, value in partitioned_values.items():
        result = scatter(value, indices, result=result, **kwargs)
    return result
