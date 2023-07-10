Design notes: ChiRho
~~~~~~~~~~~~~~~~~~~~~~~~~

The probabilistic programming language `Omega.jl <http://www.zenna.org/Omega.jl/latest/>`_
exemplifies this connection with its counterfactual semantics, but making standard probabilistic
inference methods compatible with Omega’s highly expressive
measure-theoretic semantics for conditioning remains an open problem.

In contrast, Pyro is an older, more established PPL built on top of
Python and PyTorch that already has a program transformation
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

However, this transformation is too limited to be ergonomic for most
causal inference problems of interest to practitioners. This Pyro
library explores a new programming model for causal inference
intermediate in expressivity between Pyro’s limited built-in approach
and Omega’s highly expressive one, in which interventions are defined as
operations on values within a Pyro model:

.. code:: python

   def intervene(obs: torch.Tensor, act: torch.Tensor) -> torch.Tensor:
     return act
