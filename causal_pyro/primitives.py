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
from typing import Callable, Optional, TypeVar, Union

import pyro

T = TypeVar("T")

Intervention = Union[
    Optional[T],
    Callable[[T], T],
]


@pyro.poutine.runtime.effectful(type="intervene")
def intervene(
    obs: T,
    act: Intervention[T] = None,
    *,
    event_dim: Optional[int] = None,
    name: Optional[str] = None,
    **kwargs,
) -> T:
    """
    Intervene on a value in a probabilistic program.
    """
    if callable(act) and not isinstance(act, pyro.distributions.Distribution):
        return act(obs)
    elif act is None:
        return obs
    return act
