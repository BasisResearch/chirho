This Pyro library builds on Pyro’s limited built-in support for intervention and explores a new programming model for
causal inference.

Pyro is an established PPL built on top of Python and PyTorch that already has a program transformation
``pyro.poutine.do`` for intervening on ``sample`` statements.

.. code:: python

   def model():
     x = sample("x", Normal(0, 1))
     y = sample("y", Normal(x, 1))
     return x, y

   x, y = model()
   assert x != 10  # with probability 1

   with pyro.poutine.do({"x": 10}):
     x, y = model()
   assert x == 10

However, this transformation is too limited to be ergonomic for most causal inference problems of interest to
practitioners. Instead, this library defines interventions as operations on values within a Pyro model:

.. code:: python

    def intervene(obs: torch.Tensor, act: torch.Tensor) -> torch.Tensor:
        return act

    def model():
        x = sample("x", Normal(0, 1))
        intervene(x, 10)
        y = sample("y", Normal(x, 1))
        return x, y
