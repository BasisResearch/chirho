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
    Index sets represent sets of indices of variables in a model, and sets of
    indices of observations in a dataset.

    Index sets are implemented as a dictionary mapping from strings to sets of integers.
    The strings are labels for the sets of indices, and the integers are the indices
    of the elements in the list or array.

    For example, the index set::

        {"x": {0, 1}, "y": {2, 3}}

    represents the sets of indices of the variables "x" and "y" in a model.
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
    Compute the union of multiple indexsets.

    Example::

        join(IndexSet(a={0, 1}), IndexSet(a={1, 2})) == IndexSet(a={0, 1, 2})

    .. note::
        ``join`` is associative, commutative and idempotent::

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
