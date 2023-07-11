Pyro's design makes extensive use of `algebraic effect handlers <http://pyro.ai/examples/effect_handlers.html>`_,
a technology from programming language research for representing side effects compositionally.
As described in the `Pyro introductory tutorial <http://pyro.ai/examples/intro_long.html>`_,
sampling or observing a random variable is done using the ``pyro.sample`` primitive,
whose behavior is modified by effect handlers during posterior inference.

.. code:: python

    @pyro.poutine.runtime.effectful
    def sample(name: str, dist: pyro.distributions.Distribution, obs: Optional[Tensor] = None) -> Tensor:
        return obs if obs is not None else dist.sample()

As discussed in the Introduction, Pyro already has an effect handler ``pyro.poutine.do`` for intervening on ``sample`` statements, but its implementation is too limited to be ergonomic for most causal inference problems of interest to practitioners.

The polymorphic definition of ``intervene`` above can be expanded as the generic type ``Intervention`` is made explicit.

.. code:: python

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